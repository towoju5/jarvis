"""Voice trigger hub: global hotkey + wake word + mic capture + STT/TTS.

Two ways in: the hotkey (Phase 1) or the acoustic wake word (Phase 2, see
core/wake_word.py for its accuracy caveat with a custom AGENT_NAME). Either
one starts a recording window; a simple energy-based VAD ends it on
silence. The captured audio is transcribed and published as a
TRANSCRIPT_READY event carrying the text payload.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from pynput import keyboard

from config.settings import Settings, get_settings
from core.speech_engines import SpeechEngine, build_speech_engine
from core.wake_word import FRAME_SAMPLES, SAMPLE_RATE, WakeWordDetector

logger = logging.getLogger(__name__)

VOICE_TRIGGER_ACTIVATED = "VOICE_TRIGGER_ACTIVATED"
TRANSCRIPT_READY = "TRANSCRIPT_READY"
TRANSCRIPT_EMPTY = "TRANSCRIPT_EMPTY"

# Recording / VAD tuning
MAX_RECORDING_SECONDS = 10.0
SILENCE_RMS_THRESHOLD = 250.0        # int16 RMS below this counts as silence
SPEECH_RMS_THRESHOLD = 500.0         # int16 RMS above this counts as speech onset
SILENCE_FRAMES_TO_STOP = 15          # ~1.2s of silence at 80ms/frame after speech starts


@dataclass
class VoiceEvent:
    name: str
    payload: Any = None


class HotkeyRegistrationError(RuntimeError):
    """Raised when the configured hotkey string can't be registered."""


class MicStreamError(RuntimeError):
    """Raised when the microphone input stream can't be opened."""


class VoiceHub:
    """Owns the hotkey listener, mic stream, wake word, and speech engine."""

    def __init__(self, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.event_queue = event_queue
        self._loop = loop or asyncio.get_event_loop()
        self._settings: Settings = get_settings()

        self._hotkey_listener: keyboard.GlobalHotKeys | None = None
        self._wake_word = WakeWordDetector(self._settings.agent_name)
        self._speech_engine: SpeechEngine = build_speech_engine(self._settings)

        self._raw_frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._consumer_task: asyncio.Task | None = None
        self._manual_trigger = threading.Event()
        self._recording_lock = asyncio.Lock()
        self._stopped = False

    # -- publishing -----------------------------------------------------

    def _publish(self, event_name: str, payload: Any = None) -> None:
        logger.info("voice event: %s%s", event_name, f" payload={payload!r}" if payload else "")
        self._loop.call_soon_threadsafe(self.event_queue.put_nowait, VoiceEvent(event_name, payload))

    # -- hotkey (thread callback) ----------------------------------------

    def _on_hotkey(self) -> None:
        self._publish(VOICE_TRIGGER_ACTIVATED)
        self._manual_trigger.set()

    def _start_hotkey_listener(self) -> None:
        hotkey = self._settings.trigger_hotkey
        try:
            keyboard.HotKey.parse(hotkey)
        except ValueError as exc:
            raise HotkeyRegistrationError(f"Invalid TRIGGER_HOTKEY {hotkey!r}: {exc}") from exc

        try:
            self._hotkey_listener = keyboard.GlobalHotKeys({hotkey: self._on_hotkey})
            self._hotkey_listener.start()
        except Exception as exc:
            raise HotkeyRegistrationError(
                f"Failed to register global hotkey {hotkey!r} "
                f"(on Linux this requires an X11 session with input access): {exc}"
            ) from exc
        logger.info("hotkey listener active for %r", hotkey)

    # -- mic stream (PortAudio thread callback) --------------------------

    def _on_audio_block(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        self._raw_frames.put_nowait(indata[:, 0].copy())

    def _start_mic_stream(self) -> None:
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=self._on_audio_block,
            )
            self._stream.start()
        except Exception as exc:
            raise MicStreamError(f"Failed to open microphone input stream: {exc}") from exc
        logger.info("mic stream open at %d Hz, %d-sample frames", SAMPLE_RATE, FRAME_SAMPLES)

    # -- consumer loop (runs as an asyncio task) --------------------------

    async def _consume_frames(self) -> None:
        wake_word_enabled = True
        try:
            await asyncio.to_thread(self._wake_word.load)
        except Exception:
            logger.exception("wake-word model failed to load; hotkey remains available")
            wake_word_enabled = False

        while not self._stopped:
            frame = await asyncio.to_thread(self._raw_frames.get)

            triggered = self._manual_trigger.is_set()
            if triggered:
                self._manual_trigger.clear()
            elif wake_word_enabled:
                try:
                    triggered = await asyncio.to_thread(self._wake_word.process_frame, frame)
                except Exception:
                    logger.exception("wake-word inference failed; disabling it for this session")
                    wake_word_enabled = False
                if triggered:
                    self._publish(VOICE_TRIGGER_ACTIVATED)

            if triggered:
                await self._record_and_transcribe()

    async def _record_and_transcribe(self) -> None:
        if self._recording_lock.locked():
            return
        async with self._recording_lock:
            # Drain any frames queued up while we were idle so recording
            # starts from "now", not from a stale backlog.
            while not self._raw_frames.empty():
                self._raw_frames.get_nowait()

            frames: list[np.ndarray] = []
            speech_started = False
            silent_run = 0
            start = time.monotonic()

            while time.monotonic() - start < MAX_RECORDING_SECONDS:
                frame = await asyncio.to_thread(self._raw_frames.get)
                frames.append(frame)
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

                if rms >= SPEECH_RMS_THRESHOLD:
                    speech_started = True
                    silent_run = 0
                elif speech_started:
                    silent_run += 1
                    if silent_run >= SILENCE_FRAMES_TO_STOP:
                        break

            if not speech_started:
                self._publish(TRANSCRIPT_EMPTY)
                return

            audio = np.concatenate(frames)
            text = await self._speech_engine.transcribe(audio, SAMPLE_RATE)
            if text:
                self._publish(TRANSCRIPT_READY, text)
            else:
                self._publish(TRANSCRIPT_EMPTY)

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Registers the hotkey and opens the mic stream independently --
        a failure in one (e.g. no X11 session, or PortAudio missing) does
        not prevent the other from working. Raises only if both fail.
        """
        hotkey_error: HotkeyRegistrationError | None = None
        mic_error: MicStreamError | None = None

        try:
            self._start_hotkey_listener()
        except HotkeyRegistrationError as exc:
            hotkey_error = exc
            logger.error("%s", exc)

        try:
            self._start_mic_stream()
            self._consumer_task = self._loop.create_task(self._consume_frames())
        except MicStreamError as exc:
            mic_error = exc
            logger.error("%s (mic capture, wake word, and TTS playback are disabled)", exc)

        if hotkey_error and mic_error:
            raise mic_error

    async def speak(self, text: str) -> None:
        await self._speech_engine.speak(text)

    def stop(self) -> None:
        self._stopped = True
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            self._hotkey_listener = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            self._consumer_task = None
        logger.info("voice hub stopped")
