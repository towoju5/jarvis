"""Entrypoint. Phase 1: boots settings + the hotkey trigger and reacts to
VOICE_TRIGGER_ACTIVATED events. Audio transcription (Phase 2), approval flow
(Phase 4) and publishing (Phase 5) hang off this same event loop later.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from config.settings import get_settings, settings_manager
from core.voice_hub import VOICE_TRIGGER_ACTIVATED, HotkeyRegistrationError, VoiceHub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


async def handle_event(event_name: str) -> None:
    if event_name == VOICE_TRIGGER_ACTIVATED:
        settings = get_settings()
        logger.info("%s is listening...", settings.agent_name)
        # Phase 2 wires actual audio capture + transcription in here.


async def main() -> None:
    settings = get_settings()
    logger.info("booting %s (execution_mode=%s)", settings.agent_name, settings.execution_mode)

    def on_settings_change(old, new) -> None:
        if old.agent_name != new.agent_name:
            logger.info("agent identity updated live: %r -> %r", old.agent_name, new.agent_name)

    settings_manager.subscribe(on_settings_change)
    settings_manager.start_watching()

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[str] = asyncio.Queue()
    voice_hub = VoiceHub(event_queue, loop)

    try:
        voice_hub.start()
    except HotkeyRegistrationError:
        logger.exception("could not register hotkey listener; continuing without it")

    logger.info("ready. press %s to trigger.", settings.trigger_hotkey)

    try:
        while True:
            event_name = await event_queue.get()
            await handle_event(event_name)
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
