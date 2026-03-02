"""
Detector Protocol Definition.

Protocol = structural contract. Any class with predict() and filter_detections()
methods automatically satisfies this Protocol.

Why Protocol?
- Swap different detectors (YOLO, RT-DETR, ONNX, Triton) seamlessly
- No inheritance required - structural typing
- Works with dependency injection

Example:
    >>> def process(detector: DetectorProtocol, frame):
    ...     return detector.predict(frame)
    ...
    >>> process(yolo_detector, frame)    # Works!
    >>> process(onnx_detector, frame)    # Works!
    >>> process(triton_detector, frame)  # Works!
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .detections import Detection


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Protocol defining the interface for object detectors.

    Required attributes:
        conf_threshold: Minimum confidence for detections (0.0-1.0)
        class_ids: Set of class IDs to detect (None = all classes)

    Required methods:
        predict(frame): Run inference on a single frame
        filter_detections(detections): Apply confidence and class filtering

    Example:
        >>> def run_pipeline(detector: DetectorProtocol, video_path: str):
        ...     frame = read_frame(video_path)
        ...     detections = detector.predict(frame)
        ...     return detector.filter_detections(detections)
    """

    conf_threshold: float
    class_ids: set[int] | None

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames.

        More efficient than individual predict() calls due to:
        - Single GPU kernel launch instead of N
        - Better GPU memory bandwidth utilization

        Args:
            frames: List of BGR images, each shape (H, W, 3)

        Returns:
            List of detection lists, one per frame
        """
        ...

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        """
        Filter detections by confidence and class.

        Args:
            detections: Raw detections from predict()

        Returns:
            Filtered detections with class_name added
        """
        ...
