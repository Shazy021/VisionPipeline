"""
Input size types for VisionPipeline.

Dataclasses for video metadata and input size resolution results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class VideoInfo:
    """Video metadata container."""

    original_width: int
    original_height: int
    optimal_width: int
    optimal_height: int
    fps: int
    total_frames: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInfo:
        """Create from dictionary returned by get_video_optimal_size()."""
        return cls(
            original_width=data["original_width"],
            original_height=data["original_height"],
            optimal_width=data["optimal_width"],
            optimal_height=data["optimal_height"],
            fps=data["fps"],
            total_frames=data["total_frames"],
        )


@dataclass
class InputSizeResult:
    """
    Result of input size resolution.

    Attributes:
        height: Input height (divisible by 32)
        width: Input width (divisible by 32)
        source: Where the size came from ("cli", "triton", "onnx", "auto", "fixed")
        backend_url: Backend server URL (for Triton), None otherwise
    """

    height: int
    width: int
    source: str
    backend_url: str | None = None

    @property
    def size(self) -> tuple[int, int]:
        """Return as (height, width) tuple for detector."""
        return (self.height, self.width)

    @property
    def size_hw(self) -> tuple[int, int]:
        """Return as (height, width) - alias for size."""
        return (self.height, self.width)

    @property
    def size_wh(self) -> tuple[int, int]:
        """Return as (width, height) for OpenCV resize."""
        return (self.width, self.height)


class ConfigProtocol(Protocol):
    """Protocol for config object (avoids circular import)."""

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        ...
