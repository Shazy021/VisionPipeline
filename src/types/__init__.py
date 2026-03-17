"""
Type definitions for VisionPipeline.

Exports:
- Protocols (interfaces)
- TypedDict (data structures)
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from .protocols import (
    ConfigProtocol,
    DetectorProtocol,
    VideoReaderProtocol,
    VisualizerProtocol,
)

__all__ = [
    "Detection",
    "VideoInfo",
    "ConfigProtocol",
    "DetectorProtocol",
    "VideoReaderProtocol",
    "VisualizerProtocol",
]


# =============================================================================
# TYPED DICT - Data structures
# =============================================================================


class Detection(TypedDict):
    """
    Detection structure.

    TypedDict = dictionary with known keys and types.
    Unlike regular dict, mypy/pyright check keys!
    """

    bbox: list[float]  # [x1, y1, x2, y2]
    conf: float  # Confidence 0.0-1.0
    class_id: int  # COCO class ID (0-79)
    class_name: NotRequired[str]  # Optional, added after filtering


class VideoInfo(TypedDict):
    """
    Video metadata.

    Used for VideoWriter initialization and FPS calculation.
    """

    fps: float
    width: int
    height: int
    total_frames: int | None
