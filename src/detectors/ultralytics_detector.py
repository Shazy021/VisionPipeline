"""
Ultralytics YOLO Detector Wrapper.

This module provides a unified interface for Ultralytics YOLO models.
It wraps the Ultralytics API and handles:
- Model loading (PyTorch .pt or TensorRT .engine files)
- Inference with configurable class filtering
- Batch processing with TensorRT support
- Standardized output format

Supported architectures:
- YOLO (v8, v11, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from src.detectors.base_detector import BaseDetector
from src.types import Detection

if TYPE_CHECKING:
    from ultralytics import YOLO

try:
    from ultralytics import YOLO as _YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    _YOLO = None
    logger.warning("ultralytics not installed. Run: pip install ultralytics")


class UltralyticsDetector(BaseDetector):
    """
    Unified detector for Ultralytics YOLO models.

    This class wraps the Ultralytics API to provide a consistent interface.
    It supports multiple backends implicitly based on file extension:
        - .pt files: PyTorch backend
        - .engine files: TensorRT backend (optimized inference)

    Key features:
    - Configurable class filtering
    - GPU acceleration with FP16 support
    - Batch inference for improved throughput
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        img_size: tuple[int, int] = (640, 640),
        device: str = "cuda",
        model_type: str = "yolo",  # noqa: ARG002 - kept for API compatibility
        class_ids: list[int] | None = None,
        half: bool = True,
        augment: bool = False,
        batch_size: int = 1,
        warmup_iterations: int = 3,
    ) -> None:
        """
        Initialize the Ultralytics YOLO detector.

        Args:
            model_path: Path to the model weights (.pt or .engine file).
            conf_threshold: Confidence threshold for detections.
            img_size: Input image size (height, width).
            device: Target device ('cuda' or 'cpu').
            model_type: Deprecated - kept for API compatibility.
            class_ids: List of class IDs to detect. None = all classes.
            half: Use FP16 inference (2x speedup on RTX GPUs).
            augment: Enable test-time augmentation (slower but more accurate).
            batch_size: Batch size for inference. Must match TensorRT engine.
            warmup_iterations: Number of warmup iterations for GPU.
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        super().__init__(conf_threshold, class_ids)

        self.model_path = model_path
        self.img_size = img_size
        self.device = device
        self.half = half
        self.augment = augment
        self.batch_size = batch_size

        self.is_tensorrt = model_path.endswith(".engine")

        self._model: YOLO = _YOLO(model_path, task="detect")

        if self.is_tensorrt:
            logger.success(f"[Ultralytics] TensorRT engine loaded: {model_path}")

        if warmup_iterations > 0:
            self.warmup(warmup_iterations)

        logger.info(
            f"[Ultralytics] Ready: half={half}, augment={augment}, "
            f"batch_size={batch_size}, img_size={img_size}"
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: Input image in BGR format (OpenCV standard).

        Returns:
            List of Detection dicts with bbox, conf, class_id
        """
        classes_param = list(self.class_ids) if self.class_ids else None

        results = self._model(
            frame,
            verbose=False,
            classes=classes_param,
            imgsz=self.img_size,
            device=self.device,
            half=self.half,
            augment=self.augment,
        )

        return self._parse_result(results[0])

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames.

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists, one per frame
        """
        if not frames:
            return []

        if len(frames) == 1:
            return [self.predict(frames[0])]

        n_frames = len(frames)
        classes_param = list(self.class_ids) if self.class_ids else None

        padded = False

        if self.is_tensorrt and self.batch_size > 1 and n_frames < self.batch_size:
            last_frame = frames[-1]
            frames = frames + [last_frame.copy() for _ in range(self.batch_size - n_frames)]
            padded = True

        results = self._model(
            frames,
            verbose=False,
            classes=classes_param,
            imgsz=self.img_size,
            device=self.device,
            half=self.half,
            augment=self.augment,
        )

        all_detections = [self._parse_result(r) for r in results]

        if padded:
            all_detections = all_detections[:n_frames]

        return all_detections

    def _parse_result(self, result: Any) -> list[Detection]:
        """
        Parse Ultralytics result to Detection format.

        Args:
            result: Single Ultralytics result object

        Returns:
            List of Detection dicts
        """
        detections: list[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            detection: Detection = {
                "bbox": xyxy[i].tolist(),
                "conf": float(confs[i]),
                "class_id": int(classes[i]),
            }
            detections.append(detection)

        return self.filter_detections(detections)

    def warmup(self, iterations: int = 3) -> None:
        """
        Warmup the model with dummy inference.

        Args:
            iterations: Number of warmup iterations
        """
        h, w = self.img_size

        if self.batch_size == 1:
            dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(iterations):
                self.predict(dummy_frame)
        else:
            dummy_batch = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(self.batch_size)]
            for _ in range(iterations):
                self.predict_batch(dummy_batch)

        try:
            import torch  # type: ignore[import-not-found]

            if self.device == "cuda":
                torch.cuda.synchronize()
        except ImportError:
            pass

        logger.success(
            f"[Ultralytics] Warmup complete ({iterations} iterations, batch={self.batch_size})"
        )
