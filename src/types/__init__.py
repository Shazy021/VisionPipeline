"""
Types module for VisionPipeline.

Provides type-safe structures for detection results and detector interfaces.
"""

from .detections import Detection, DetectionList
from .protocols import DetectorProtocol

__all__ = ["Detection", "DetectionList", "DetectorProtocol"]
