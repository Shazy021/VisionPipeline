"""
Protocol definitions for VisionPipeline.

Protocols = contracts (interfaces) in Python.
They define what methods a class MUST have, but don't require inheritance.

Benefits:
1. Type safety - mypy/pyright check compliance
2. Flexibility - any class with required methods works
3. Documentation - clear expectations

Example:
    class MyConfig:
        def get(self, key: str, default: Any = None) -> Any:
            return "value"

    # MyConfig automatically satisfies ConfigProtocol!
    # Because it has the get() method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from src.types import Detection


# =============================================================================
# CONFIG PROTOCOL
# =============================================================================


@runtime_checkable
class ConfigProtocol(Protocol):
    """
    Configuration contract.

    Any class with get() and get_model_path() methods
    automatically satisfies this protocol.
    """

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get value by dot notation.

        Args:
            key_path: Path to value, e.g. "models.default_model"
            default: Default value if key not found

        Returns:
            Value from config or default
        """
        ...

    def get_model_path(self, model: str, backend: str) -> str:
        """
        Get model weights path.

        Args:
            model: Model name (yolo, etc.)
            backend: Backend (pytorch, onnx, tensorrt, triton)

        Returns:
            Path to weights file

        Raises:
            KeyError: If path not found in config
        """
        ...


# =============================================================================
# DETECTOR PROTOCOL
# =============================================================================


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Object detector contract.

    Any detector (ONNX, Triton, Ultralytics) must satisfy this protocol.
    This enables functions that work with ANY detector.
    """

    conf_threshold: float
    class_ids: set[int] | None

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            List of detections
        """
        ...

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames.

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists for each frame
        """
        ...

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        """
        Filter detections by confidence and class.

        Args:
            detections: Raw detections from predict()

        Returns:
            Filtered detections with class_name
        """
        ...


# =============================================================================
# VIDEO READER PROTOCOL
# =============================================================================


@runtime_checkable
class VideoReaderProtocol(Protocol):
    """
    Video reader contract.

    Enables swapping video sources (file, RTSP, camera)
    without changing pipeline code.
    """

    def read_frame(self) -> np.ndarray | None:
        """
        Read next frame.

        Returns:
            BGR frame or None (end of video)
        """
        ...

    def get_info(self) -> dict[str, Any]:
        """
        Get video metadata.

        Returns:
            {
                'fps': 30,
                'width': 1920,
                'height': 1080,
                'total_frames': 1000
            }
        """
        ...

    def release(self) -> None:
        """Release resources."""
        ...


# =============================================================================
# VISUALIZER PROTOCOL
# =============================================================================


@runtime_checkable
class VisualizerProtocol(Protocol):
    """
    Detection visualization contract.

    Enables changing drawing style (colors, fonts, layout)
    without changing pipeline logic.
    """

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw detections on frame.

        Args:
            frame: Source frame
            detections: List of detections

        Returns:
            Frame with bounding boxes drawn
        """
        ...

    def draw_info(
        self,
        frame: np.ndarray,
        fps: float,
        frame_count: int,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draw info panel (FPS, counter, legend).

        Args:
            frame: Frame with detections
            fps: Current FPS
            frame_count: Frame number
            detections: Detections for class counting

        Returns:
            Frame with info panel
        """
        ...
