"""
Ultralytics Detector Wrapper.

This module provides a unified interface for Ultralytics models (YOLO, RT-DETR).
It wraps the Ultralytics API and handles:
- Model loading (PyTorch .pt or TensorRT .engine files)
- Inference with configurable class filtering
- Standardized output format

Supported architectures:
- YOLO (v8, v11, etc.)
- RT-DETR (Real-Time DEtection Transformer)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ultralytics import RTDETR, YOLO

from src.detectors.base_detector import BaseDetector
from src.types import Detection

if TYPE_CHECKING:
    pass


class UltralyticsDetector(BaseDetector):
    """
    Unified detector for Ultralytics models (YOLO and RT-DETR).

    This class wraps the Ultralytics API to provide a consistent interface.
    It supports multiple backends implicitly based on file extension:
        - .pt files: PyTorch backend
        - .engine files: TensorRT backend (optimized inference)

    Key features:
    - Automatic model type detection
    - Configurable class filtering
    - GPU acceleration support
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        img_size: tuple[int, int] = (640, 640),
        device: str = "cuda",
        model_type: str = "yolo",
        class_ids: list[int] | None = None,
    ):
        """
        Initialize the Ultralytics detector.

        Args:
            model_path: Path to the model weights (.pt or .engine file).
            conf_threshold: Confidence threshold for detections.
            img_size: Input image size (height, width).
            device: Target device ('cuda' or 'cpu').
            model_type: Type of model architecture ('yolo' or 'rtdetr').
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

        self.model_type = model_type.lower()
        self.model_path = model_path
        self.img_size = img_size
        self.device = device

        # Load model based on type
        if self.model_type == "rtdetr":
            self._model: YOLO | RTDETR = RTDETR(model_path)
        else:
            self._model = YOLO(model_path)

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
