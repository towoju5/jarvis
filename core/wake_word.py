"""Acoustic wake-word detection via openWakeWord.

Important limitation: openWakeWord ships fixed, pretrained keyword models
(e.g. "hey_jarvis", "alexa", "hey_mycroft"). There is no way to retrain the
acoustic model on the fly when AGENT_NAME changes at runtime -- the live
.env reload updates the agent's *spoken identity* and how it refers to
itself, but the wake-word audio model stays pinned to whichever bundled
model was loaded at startup. If AGENT_NAME doesn't map to a bundled model,
we fall back to "hey_jarvis" and log a warning; the hotkey trigger remains
the reliable path for a custom name.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # openWakeWord expects ~80ms chunks at 16kHz

# Bundled openWakeWord models we can map a spoken AGENT_NAME onto.
_BUNDLED_MODELS = {
    "jarvis": "hey_jarvis",
    "alexa": "alexa",
    "mycroft": "hey_mycroft",
}
_DEFAULT_MODEL = "hey_jarvis"


def resolve_model_name(agent_name: str) -> str:
    key = agent_name.strip().lower()
    if key in _BUNDLED_MODELS:
        return _BUNDLED_MODELS[key]
    logger.warning(
        "no bundled openWakeWord model matches AGENT_NAME=%r; falling back to %r. "
        "Acoustic wake-word detection will not actually respond to %r -- "
        "use the hotkey trigger for a custom name.",
        agent_name, _DEFAULT_MODEL, agent_name,
    )
    return _DEFAULT_MODEL


class WakeWordDetector:
    def __init__(self, agent_name: str, threshold: float = 0.5) -> None:
        self.model_name = resolve_model_name(agent_name)
        self.threshold = threshold
        self._model = None

    def load(self) -> None:
        """Blocking load -- call via asyncio.to_thread, not on the event loop."""
        from openwakeword import utils
        from openwakeword.model import Model

        try:
            utils.download_models(model_names=[self.model_name])
        except Exception:
            logger.warning(
                "could not fetch/verify openWakeWord model %r (offline?); "
                "will use a cached copy if one exists", self.model_name,
            )

        self._model = Model(wakeword_models=[self.model_name])
        logger.info("wake-word model %r loaded", self.model_name)

    def process_frame(self, frame: np.ndarray) -> bool:
        """frame: int16 mono PCM, length FRAME_SAMPLES. Returns True on detection."""
        if self._model is None:
            raise RuntimeError("WakeWordDetector.load() must be called before process_frame()")
        predictions = self._model.predict(frame)
        score = predictions.get(self.model_name, 0.0)
        return score >= self.threshold
