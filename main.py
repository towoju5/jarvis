"""Entrypoint. Phase 1: boots settings + the hotkey trigger and reacts to
VOICE_TRIGGER_ACTIVATED events. Audio transcription (Phase 2), approval flow
(Phase 4) and publishing (Phase 5) hang off this same event loop later.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from config.settings import get_settings, settings_manager
from core.voice_hub import (
    TRANSCRIPT_EMPTY,
    TRANSCRIPT_READY,
    VOICE_TRIGGER_ACTIVATED,
    HotkeyRegistrationError,
    MicStreamError,
    VoiceEvent,
    VoiceHub,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


async def handle_event(event: VoiceEvent, voice_hub: VoiceHub) -> None:
    settings = get_settings()
    if event.name == VOICE_TRIGGER_ACTIVATED:
        logger.info("%s is listening...", settings.agent_name)
    elif event.name == TRANSCRIPT_READY:
        text = event.payload
        logger.info("heard: %r", text)
        # Phase 4/5 route this to the state manager + approval flow instead
        # of a canned reply.
        await voice_hub.speak(f"You said: {text}")
    elif event.name == TRANSCRIPT_EMPTY:
        logger.info("no speech detected")


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

    logger.info("ready. press %s or say the wake word to trigger.", settings.trigger_hotkey)

    try:
        while True:
            event = await event_queue.get()
            await handle_event(event, voice_hub)
    except asyncio.CancelledError:
        pass
    finally:
        voice_hub.stop()
        settings_manager.stop_watching()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("shutting down")
