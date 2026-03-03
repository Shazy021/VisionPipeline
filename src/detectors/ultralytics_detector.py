"""
Ultralytics YOLO Detector Wrapper.

This module provides a unified interface for Ultralytics YOLO models.
It wraps the Ultralytics API and handles:
- Model loading (PyTorch .pt or TensorRT .engine files)
- Inference with configurable class filtering
- Standardized output format

Supported architectures:
- YOLO (v8, v11, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ultralytics import YOLO

from src.detectors.base_detector import BaseDetector
from src.types import Detection

if TYPE_CHECKING:
    pass


class UltralyticsDetector(BaseDetector):
    """
    Unified detector for Ultralytics YOLO models.

    This class wraps the Ultralytics API to provide a consistent interface.
    It supports multiple backends implicitly based on file extension:
        - .pt files: PyTorch backend
        - .engine files: TensorRT backend (optimized inference)

    Key features:
    - Configurable class filtering
    - GPU acceleration support
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        img_size: tuple[int, int] = (640, 640),
        device: str = "cuda",
        model_type: str = "yolo",  # noqa: ARG002 - kept for API compatibility
        class_ids: list[int] | None = None,
    ):
        """
        Initialize the Ultralytics YOLO detector.

        Args:
            model_path: Path to the model weights (.pt or .engine file).
            conf_threshold: Confidence threshold for detections.
            img_size: Input image size (height, width).
            device: Target device ('cuda' or 'cpu').
            model_type: Deprecated - kept for API compatibility. Only YOLO is supported.
            class_ids: List of class IDs to detect. None = all classes.

        Example:
            >>> # Detect all classes
            >>> detector = UltralyticsDetector("yolo11n.pt")

            >>> # Detect only persons
            >>> detector = UltralyticsDetector("yolo11n.pt", class_ids=[0])

            >>> # Detect persons and vehicles
            >>> detector = UltralyticsDetector("yolo11n.pt", class_ids=[0, 2, 3, 5, 7])
        """
        # Initialize base class
        super().__init__(conf_threshold, class_ids)

        self.model_path = model_path
        self.img_size = img_size
        self.device = device

        # Load YOLO model
        self._model: YOLO = YOLO(model_path)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame.

        The Ultralytics model automatically handles backend selection
        (PyTorch or TensorRT) based on the loaded file type.

        Args:
            frame: Input image in BGR format (OpenCV standard).

        Returns:
            List of Detection dicts with bbox, conf, class_id, class_name
        """
        # Convert class_ids set to list for Ultralytics API
        # Note: Ultralytics expects 'classes' parameter as list
        classes_param = list(self.class_ids) if self.class_ids else None

        # Run inference
        results = self._model(
            frame,
            verbose=False,
            classes=classes_param,  # Filter classes at model level
            imgsz=self.img_size,
            device=self.device,
        )

        # Parse results
        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get bounding box coordinates (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()

                    # Get confidence score
                    conf = float(box.conf[0].cpu().numpy())

                    # Get class ID
                    cls = int(box.cls[0].cpu().numpy())

                    detection: Detection = {"bbox": xyxy.tolist(), "conf": conf, "class_id": cls}
                    detections.append(detection)

        # Apply additional filtering and add class names
        return self.filter_detections(detections)

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames.

        Ultralytics natively supports batch inference - just pass
        all frames at once for optimized GPU utilization.

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists, one per frame
        """
        if not frames:
            return []

        # Single frame fallback
        if len(frames) == 1:
            return [self.predict(frames[0])]

        classes_param = list(self.class_ids) if self.class_ids else None

        # Run batch inference - Ultralytics handles this natively
        results = self._model(
            frames, verbose=False, classes=classes_param, imgsz=self.img_size, device=self.device
        )

        # Parse results for each frame
        all_detections: list[list[Detection]] = []

        for r in results:
            frame_detections: list[Detection] = []

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    detection: Detection = {"bbox": xyxy.tolist(), "conf": conf, "class_id": cls}
                    frame_detections.append(detection)

            all_detections.append(self.filter_detections(frame_detections))

        return all_detections

    def warmup(self, iterations: int = 3) -> None:
        """
        Warmup the model with dummy inference.

        This is useful for:
        - Loading model to GPU memory
        - Optimizing CUDA kernels
        - Getting accurate timing on first real inference

        Args:
            iterations: Number of warmup iterations
        """
        dummy_frame = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        for _ in range(iterations):
            self.predict(dummy_frame)
