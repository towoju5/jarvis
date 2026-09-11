"""Entrypoint: wires the voice trigger through either general conversation
or the video-generation/approval/publish pipeline, chosen by intent.

Only explicit "make/create/generate a video/clip/anime ..." phrasing
routes to media_generator -> Telegram approval -> YouTube upload (see
SUPPORTED.md). Everything else goes to core/chat_engine.py for a spoken
conversational reply. A NotImplementedError from an unconfigured backend
(no video-gen provider chosen, no platform credentials set, no chat
provider API key set) is handled as "not set up yet" -- it's not a bug
the watchdog can patch, so it's reported and the loop continues. Any
other exception is left to propagate so it reaches logs/runtime.log for
core/watchdog.py to catch.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys

from communication.notify_bridge import TelegramApprovalBridge
from config.settings import get_settings, settings_manager
from core.chat_engine import ChatEngine, build_chat_engine
from core.dashboard import AgentStatus, LogBuffer, create_app, start_dashboard
from core.state_manager import StateManager
from core.voice_hub import (
    TRANSCRIPT_EMPTY,
    TRANSCRIPT_READY,
    VOICE_TRIGGER_ACTIVATED,
    HotkeyRegistrationError,
    MicStreamError,
    VoiceEvent,
    VoiceHub,
)
from modules.media_generator import AnimeVideoGenerator, GenerationRequest, MediaGenerator
from modules.social_poster import YouTubeClient

_VIDEO_REQUEST_RE = re.compile(
    r"\b(make|create|generate|produce)\b.{0,25}\b(video|clip|anime|animation)\b", re.IGNORECASE
)


def _is_video_request(text: str) -> bool:
    return bool(_VIDEO_REQUEST_RE.search(text))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

log_buffer = LogBuffer()
logging.getLogger().addHandler(log_buffer)


async def handle_video_request(
    text: str,
    voice_hub: VoiceHub,
    state_manager: StateManager,
    media_generator: MediaGenerator,
    youtube_client: YouTubeClient,
    status: AgentStatus,
) -> None:
    task = state_manager.create_task(text)
    status.state, status.detail = "generating", text

    try:
        result = await media_generator.generate(GenerationRequest(prompt=text))
    except NotImplementedError as exc:
        logger.warning("task %s: media generation not configured: %s", task.id, exc)
        state_manager.mark_failed(task, str(exc))
        await voice_hub.speak("Media generation isn't set up yet.")
        status.state, status.detail = "idle", ""
        return

    status.state = "pending_approval"
    summary = f"{result.title or text}\n\nApprove publishing this to YouTube?"
    approved = await state_manager.request_publish_approval(task, summary)
    if not approved:
        await voice_hub.speak("Not approved. Discarding.")
        status.state, status.detail = "idle", ""
        return

    status.state = "publishing"
    upload = await youtube_client.upload(result.video_path, result.title, result.description)
    state_manager.mark_done(task, upload)
    await voice_hub.speak(f"Published to YouTube: {upload.url}")
    status.state, status.detail = "idle", ""


async def handle_chat(
    text: str,
    voice_hub: VoiceHub,
    state_manager: StateManager,
    chat_engine: ChatEngine,
    status: AgentStatus,
    agent_name: str,
) -> None:
    task = state_manager.create_task(text)
    status.state, status.detail = "generating", text

    try:
        reply = await chat_engine.respond(agent_name, text)
    except Exception as exc:
        logger.warning("task %s: chat reply failed: %s", task.id, exc)
        state_manager.mark_failed(task, str(exc))
        await voice_hub.speak("I couldn't reach any AI provider just now.")
        status.state, status.detail = "idle", ""
        return

    state_manager.mark_done(task, reply)
    await voice_hub.speak(reply)
    status.state, status.detail = "idle", ""


async def handle_event(
    event: VoiceEvent,
    voice_hub: VoiceHub,
    state_manager: StateManager,
    media_generator: MediaGenerator,
    youtube_client: YouTubeClient,
    chat_engine: ChatEngine,
    status: AgentStatus,
) -> None:
    settings = get_settings()
    if event.name == VOICE_TRIGGER_ACTIVATED:
        logger.info("%s is listening...", settings.agent_name)
        status.state, status.detail = "listening", ""
    elif event.name == TRANSCRIPT_READY:
        text = event.payload
        logger.info("heard: %r", text)
        if _is_video_request(text):
            await handle_video_request(text, voice_hub, state_manager, media_generator, youtube_client, status)
        else:
            await handle_chat(text, voice_hub, state_manager, chat_engine, status, settings.agent_name)
    elif event.name == TRANSCRIPT_EMPTY:
        logger.info("no speech detected")
        status.state, status.detail = "idle", ""


async def main() -> None:
    settings = get_settings()
    logger.info("booting %s (execution_mode=%s)", settings.agent_name, settings.execution_mode)

    def on_settings_change(old, new) -> None:
        if old.agent_name != new.agent_name:
            logger.info("agent identity updated live: %r -> %r", old.agent_name, new.agent_name)

    settings_manager.subscribe(on_settings_change)
    settings_manager.start_watching()

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[VoiceEvent] = asyncio.Queue()
    voice_hub = VoiceHub(event_queue, loop)

    try:
        voice_hub.start()
    except (HotkeyRegistrationError, MicStreamError):
        logger.exception("could not start any voice trigger (hotkey and mic both failed)")

    approval_bridge: TelegramApprovalBridge | None = None
    if settings.telegram_bot_token and settings.telegram_chat_id:
        approval_bridge = TelegramApprovalBridge(settings.telegram_bot_token, settings.telegram_chat_id)
        try:
            await approval_bridge.start()
        except Exception:
            logger.exception("failed to start telegram approval bridge; publishing will stay unapproved")
            approval_bridge = None
    else:
        logger.warning("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; publish approval will always be rejected")

    state_manager = StateManager(approval_bridge)
    media_generator = AnimeVideoGenerator()
    youtube_client = YouTubeClient()
    chat_engine = build_chat_engine(settings)
    status = AgentStatus()

    # Dashboard comes up before the (potentially slow, first-run) speech
    # model warmup below -- it must not be blocked behind that, since it's
    # exactly what you'd want to check *during* a slow warmup.
    dashboard_app = create_app(get_settings, status, state_manager, log_buffer)
    dashboard_runner = await start_dashboard(dashboard_app, "127.0.0.1", settings.dashboard_port)
    logger.info("dashboard: http://127.0.0.1:%d/", settings.dashboard_port)

    async def _run_warmup() -> None:
        status.state, status.detail = "warming_up", "loading speech models"
        logger.info("loading speech models in the background (first run downloads them)...")
        await voice_hub.warmup()
        if status.state == "warming_up":
            status.state, status.detail = "idle", ""

    warmup_task = asyncio.create_task(_run_warmup())

    if settings.enable_hotkey_trigger:
        logger.info("ready. press %s or say the wake word to trigger.", settings.trigger_hotkey)
    else:
        logger.info("ready. say the wake word to trigger (hotkey disabled -- see ENABLE_HOTKEY_TRIGGER).")

    try:
        while True:
            event = await event_queue.get()
            await handle_event(event, voice_hub, state_manager, media_generator, youtube_client, chat_engine, status)
    except asyncio.CancelledError:
        pass
    finally:
        warmup_task.cancel()
        voice_hub.stop()
        settings_manager.stop_watching()
        if approval_bridge is not None:
            await approval_bridge.stop()
        await dashboard_runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutting down")
