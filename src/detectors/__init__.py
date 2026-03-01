"""
Detectors module for VisionPipeline.

Provides detector implementations for various inference backends:
- UltralyticsDetector: PyTorch and TensorRT via Ultralytics API
- ONNXDetector: ONNX Runtime (CPU and CUDA)
- TritonDetector: NVIDIA Triton Inference Server

Backend Registry:
- BackendRegistry: Query backend capabilities and input size requirements
"""

from .base_detector import BaseDetector
from .onnx_detector import ONNXDetector
from .registry import BackendRegistry
from .ultralytics_detector import UltralyticsDetector

__all__ = [
    "BaseDetector",
    "UltralyticsDetector",
    "ONNXDetector",
    "BackendRegistry",
]
