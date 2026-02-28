"""
Detectors module for VisionPipeline.

Provides detector implementations for various inference backends:
- UltralyticsDetector: PyTorch and TensorRT via Ultralytics API
- ONNXDetector: ONNX Runtime (CPU and CUDA)
- TritonDetector: NVIDIA Triton Inference Server (lazy import in factory)
"""

from .base_detector import BaseDetector
from .onnx_detector import ONNXDetector
from .ultralytics_detector import UltralyticsDetector

__all__ = [
    "BaseDetector",
    "UltralyticsDetector",
    "ONNXDetector",
]
# Note: TritonDetector is NOT imported here to avoid loading tritonclient.
# It's lazily imported in factory.py when backend="triton" is requested.
