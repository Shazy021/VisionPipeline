"""
Detector Factory.

This module provides a factory pattern for creating detector instances.
The factory abstracts the creation logic, allowing the caller to specify
the model and backend without knowing the implementation details.

Supported backends:
- pytorch: Native PyTorch via Ultralytics API
- tensorrt: TensorRT engine via Ultralytics API
- onnx: ONNX Runtime
- triton: NVIDIA Triton Inference Server

Usage:
    detector = DetectorFactory.create(
        model="yolo",
        backend="pytorch",
        weights_path="weights/yolo11n.pt",
        conf_threshold=0.25
    )
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.detectors.ultralytics_detector import UltralyticsDetector

if TYPE_CHECKING:
    from src.types import DetectorProtocol


class DetectorFactory:
    """Factory class to create detector instances dynamically."""

    @staticmethod
    def create(
        model: str,
        backend: str,
        weights_path: str,
        conf_threshold: float,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        use_gpu: bool = True,
        class_ids: list[int] | None = None,
    ) -> DetectorProtocol:
        """
        Create a detector instance based on model and backend type.

        Args:
            model: Model architecture ('yolo' or 'rtdetr')
            backend: Inference backend ('pytorch', 'onnx', 'tensorrt', 'triton')
            weights_path: Path to model weights
            conf_threshold: Confidence threshold (0.0-1.0)
            nms_threshold: NMS threshold for ONNX/Triton (0.0-1.0)
            input_size: Tuple (height, width) for model input
            use_gpu: Enable GPU acceleration
            class_ids: List of class IDs to detect. None = all classes

        Returns:
            Initialized detector instance implementing DetectorProtocol

        Raises:
            ValueError: If model/backend combination is not supported
            ImportError: If required backend dependencies are not installed

        Example:
            >>> # Detect all classes
            >>> detector = DetectorFactory.create(
            ...     model="yolo", backend="pytorch",
            ...     weights_path="yolo11n.pt", conf_threshold=0.25
            ... )

            >>> # Detect only persons
            >>> detector = DetectorFactory.create(
            ...     model="yolo", backend="pytorch",
            ...     weights_path="yolo11n.pt", conf_threshold=0.25,
            ...     class_ids=[0]
            ... )
        """
        model = model.lower()
        backend = backend.lower()
        actual_device = "cuda" if use_gpu else "cpu"

        # =========================================================================
        # PyTorch / TensorRT Backend (via Ultralytics)
        # =========================================================================
        if backend in ("pytorch", "tensorrt"):
            return UltralyticsDetector(
                model_path=weights_path,
                conf_threshold=conf_threshold,
                img_size=input_size,
                device=actual_device,
                model_type=model,
                class_ids=class_ids,
            )

        # =========================================================================
        # ONNX Runtime Backend
        # =========================================================================
        elif backend == "onnx":
            from src.detectors.onnx_detector import ONNXDetector

            return ONNXDetector(
                weights_path,
                use_gpu=use_gpu,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                model_type=model,
                input_size=input_size,
                class_ids=class_ids,
            )

        # =========================================================================
        # Triton Inference Server Backend
        # =========================================================================
        elif backend == "triton":
            from src.detectors.triton_detector import TritonDetector

            url = os.getenv("TRITON_URL", "localhost:8000")

            return TritonDetector(
                model_name=model,
                url=url,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                input_size=input_size,
                class_ids=class_ids,
            )

        raise ValueError(f"Unsupported configuration: Model={model}, Backend={backend}")
