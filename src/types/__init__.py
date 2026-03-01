"""
Types module for VisionPipeline.

Provides type-safe structures for detection results, detector interfaces,
and input size resolution.
"""

from .detections import Detection, DetectionList
from .input_size import ConfigProtocol, InputSizeResult, VideoInfo
from .protocols import DetectorProtocol

__all__ = [
    "Detection",
    "DetectionList",
    "DetectorProtocol",
    "VideoInfo",
    "InputSizeResult",
    "ConfigProtocol",
]
