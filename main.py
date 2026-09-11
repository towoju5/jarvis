"""Entrypoint: wires the voice trigger through generation, approval, and
publishing. Each transcript becomes a task: generate -> request Telegram
approval -> publish to YouTube (the only platform with a real upload path
today, see SUPPORTED.md). A NotImplementedError from an unconfigured
backend (no video-gen provider chosen, no platform credentials set) is
handled as "not set up yet" -- it's not a bug the watchdog can patch, so
it's reported and the loop continues. Any other exception is left to
propagate so it reaches logs/runtime.log for core/watchdog.py to catch.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from communication.notify_bridge import TelegramApprovalBridge
from config.settings import get_settings, settings_manager
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

log_buffer = LogBuffer()
logging.getLogger().addHandler(log_buffer)


async def handle_transcript(
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


async def handle_event(
    event: VoiceEvent,
    voice_hub: VoiceHub,
    state_manager: StateManager,
    media_generator: MediaGenerator,
    youtube_client: YouTubeClient,
    status: AgentStatus,
) -> None:
    settings = get_settings()
    if event.name == VOICE_TRIGGER_ACTIVATED:
        logger.info("%s is listening...", settings.agent_name)
        status.state, status.detail = "listening", ""
    elif event.name == TRANSCRIPT_READY:
        text = event.payload
        logger.info("heard: %r", text)
        await handle_transcript(text, voice_hub, state_manager, media_generator, youtube_client, status)
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
    status = AgentStatus()

    dashboard_app = create_app(get_settings, status, state_manager, log_buffer)
    dashboard_runner = await start_dashboard(dashboard_app, "127.0.0.1", settings.dashboard_port)
    logger.info("dashboard: http://127.0.0.1:%d/", settings.dashboard_port)

    logger.info("ready. press %s or say the wake word to trigger.", settings.trigger_hotkey)

    try:
        while True:
            event = await event_queue.get()
            await handle_event(event, voice_hub, state_manager, media_generator, youtube_client, status)
    except asyncio.CancelledError:
        pass
    finally:
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
