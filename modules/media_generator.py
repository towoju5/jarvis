"""Video/image generation pipeline -- abstract by design.

No video-gen backend is pinned here: "anime video generation" could mean
a local Stable Diffusion + AnimateDiff pipeline, a paid API (Runway,
Pika, Kling, ...), or something else entirely, and that choice has real
cost/quality/licensing tradeoffs the blueprint doesn't settle. Implement
`AnimeVideoGenerator.generate()` against whichever backend you pick; every
other module (social_poster, the main loop) only depends on this
interface, so swapping backends later doesn't ripple outward.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_MEDIA_DIR = PROJECT_ROOT / "assets" / "generated"


@dataclass
class GenerationRequest:
    prompt: str
    duration_seconds: float = 5.0
    aspect_ratio: str = "9:16"  # short-form vertical default for TikTok/Reels/Shorts


@dataclass
class GenerationResult:
    video_path: Path
    thumbnail_path: Path | None = None
    title: str = ""
    description: str = ""


class MediaGenerator(abc.ABC):
    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce a video (and optionally a thumbnail) for `request`."""


class AnimeVideoGenerator(MediaGenerator):
    """Placeholder -- wire your chosen video-gen backend's API call here."""

    def __init__(self, **backend_config: str) -> None:
        self._backend_config = backend_config

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        GENERATED_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError(
            "No video generation backend is configured. Implement "
            "AnimeVideoGenerator.generate() in modules/media_generator.py "
            "against your chosen provider (local pipeline or paid API)."
        )
