"""STT/TTS engines behind a single interface, swapped by EXECUTION_MODE.

OfflineSpeechEngine: Faster-Whisper (CPU, int8) for STT, Piper for TTS.
OnlineSpeechEngine: placeholder -- the blueprint doesn't pin a vendor, so
this raises a clear error until a provider is wired in, rather than
guessing an API that may not match what you actually want to pay for.
"""
from __future__ import annotations

import abc
import asyncio
import logging
from pathlib import Path

import numpy as np

from config.settings import Settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPER_VOICES_DIR = PROJECT_ROOT / "assets" / "piper_voices"


class SpeechEngine(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """audio: mono float32 or int16 PCM. Returns transcribed text."""

    @abc.abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesize and play `text` through the default output device."""


def _resolve_piper_model_path(voice_name: str) -> Path:
    """Find (or fetch) the .onnx + .onnx.json pair for `voice_name`.

    Uses Piper's own download machinery (piper.download), which pulls from
    the rhasspy/piper-voices manifest -- the same thing the `piper` CLI's
    --download_dir flag drives.
    """
    from piper.download import VoiceNotFoundError, ensure_voice_exists, find_voice, get_voices

    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    data_dirs = [str(PIPER_VOICES_DIR)]

    try:
        onnx_path, _config_path = find_voice(voice_name, data_dirs)
        return onnx_path
    except VoiceNotFoundError:
        pass

    logger.info("piper voice %r not found locally, downloading to %s", voice_name, PIPER_VOICES_DIR)
    try:
        voices_info = get_voices(str(PIPER_VOICES_DIR), update_voices=True)
        ensure_voice_exists(voice_name, data_dirs, str(PIPER_VOICES_DIR), voices_info)
        onnx_path, _config_path = find_voice(voice_name, data_dirs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not obtain Piper voice model {voice_name!r} "
            f"(tried downloading to {PIPER_VOICES_DIR}): {exc}. "
            f"You can manually place {voice_name}.onnx and {voice_name}.onnx.json there."
        ) from exc
    return onnx_path


class OfflineSpeechEngine(SpeechEngine):
    def __init__(self, whisper_model_size: str, piper_voice_model: str) -> None:
        self._whisper_model_size = whisper_model_size
        self._piper_voice_model = piper_voice_model
        self._whisper = None
        self._piper_voice = None

    async def _ensure_whisper(self):
        if self._whisper is None:
            def _load():
                from faster_whisper import WhisperModel
                logger.info("loading faster-whisper model=%r (cpu, int8)", self._whisper_model_size)
                return WhisperModel(self._whisper_model_size, device="cpu", compute_type="int8")
            self._whisper = await asyncio.to_thread(_load)
        return self._whisper

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        model = await self._ensure_whisper()

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        def _run() -> str:
            segments, _info = model.transcribe(audio, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()

        return await asyncio.to_thread(_run)

    async def _ensure_piper(self):
        if self._piper_voice is None:
            def _load():
                from piper import PiperVoice
                model_path = _resolve_piper_model_path(self._piper_voice_model)
                logger.info("loading piper voice from %s", model_path)
                return PiperVoice.load(str(model_path))
            self._piper_voice = await asyncio.to_thread(_load)
        return self._piper_voice

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        voice = await self._ensure_piper()

        def _synthesize_and_play() -> None:
            import sounddevice as sd
            chunks = [np.frombuffer(c, dtype=np.int16) for c in voice.synthesize_stream_raw(text)]
            if not chunks:
                return
            pcm = np.concatenate(chunks)
            sd.play(pcm, samplerate=voice.config.sample_rate)
            sd.wait()

        await asyncio.to_thread(_synthesize_and_play)


class OnlineSpeechEngine(SpeechEngine):
    """Wire a remote STT/TTS provider's API calls into these two methods.

    Both are async and match OfflineSpeechEngine's signatures exactly, so
    flipping EXECUTION_MODE=ONLINE in .env needs no changes anywhere else
    in the pipeline once this is implemented.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        raise NotImplementedError(
            "EXECUTION_MODE=ONLINE has no STT provider configured. "
            "Implement OnlineSpeechEngine.transcribe() in core/speech_engines.py."
        )

    async def speak(self, text: str) -> None:
        raise NotImplementedError(
            "EXECUTION_MODE=ONLINE has no TTS provider configured. "
            "Implement OnlineSpeechEngine.speak() in core/speech_engines.py."
        )


def build_speech_engine(settings: Settings) -> SpeechEngine:
    if settings.execution_mode == "ONLINE":
        return OnlineSpeechEngine(settings)
    return OfflineSpeechEngine(settings.whisper_model_size, settings.piper_voice_model)
