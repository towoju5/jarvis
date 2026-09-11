"""Voice trigger hub.

Phase 1 scope: a background hotkey listener that fires a
VOICE_TRIGGER_ACTIVATED event onto an asyncio queue, bridging pynput's own
listener thread into the asyncio loop. The wake-word audio pipeline
(Faster-Whisper / Piper) is added in Phase 2.
"""
from __future__ import annotations

import asyncio
import logging

from pynput import keyboard

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

VOICE_TRIGGER_ACTIVATED = "VOICE_TRIGGER_ACTIVATED"


class HotkeyRegistrationError(RuntimeError):
    """Raised when the configured hotkey string can't be registered."""


class VoiceHub:
    """Owns the global hotkey listener and publishes trigger events.

    Events are pushed onto `event_queue` (an asyncio.Queue) so consumers on
    the main event loop can `await event_queue.get()` without touching
    threading directly.
    """

    def __init__(self, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.event_queue = event_queue
        self._loop = loop or asyncio.get_event_loop()
        self._listener: keyboard.GlobalHotKeys | None = None
        self._settings: Settings = get_settings()

    def _publish(self, event_name: str) -> None:
        logger.info("voice trigger fired: %s", event_name)
        self._loop.call_soon_threadsafe(self.event_queue.put_nowait, event_name)

    def _on_hotkey(self) -> None:
        self._publish(VOICE_TRIGGER_ACTIVATED)

    def start(self) -> None:
        """Register the global hotkey listener in a background thread."""
        hotkey = self._settings.trigger_hotkey
        try:
            parsed = keyboard.HotKey.parse(hotkey)
        except ValueError as exc:
            raise HotkeyRegistrationError(f"Invalid TRIGGER_HOTKEY {hotkey!r}: {exc}") from exc

        try:
            self._listener = keyboard.GlobalHotKeys({hotkey: self._on_hotkey})
            self._listener.start()
        except Exception as exc:  # pynput surfaces platform-specific errors here
            raise HotkeyRegistrationError(
                f"Failed to register global hotkey {hotkey!r} "
                f"(on Linux this requires an X11 session with input access): {exc}"
            ) from exc

        logger.info("hotkey listener active for %r (parsed as %s)", hotkey, parsed)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("hotkey listener stopped")
