"""
Detection types for VisionPipeline.

TypedDict provides type-safe dictionary structures with:
- IDE autocomplete for keys
- Static type checking with mypy
- Self-documenting code

Example:
    >>> det: Detection = {
    ...     "bbox": [100.0, 200.0, 300.0, 400.0],
    ...     "conf": 0.95,
    ...     "class_id": 0,
    ...     "class_name": "person"
    ... }
"""

from __future__ import annotations

from typing import NotRequired, Required, TypedDict


class Detection(TypedDict, total=False):
    """
    Single object detection result.

    Required fields:
        bbox: [x1, y1, x2, y2] - Bounding box in absolute pixel coordinates
        conf: float - Confidence score from 0.0 to 1.0
        class_id: int - COCO class ID (0-79)

    Optional fields:
        class_name: str - Human-readable class name (e.g., "person", "car")

    Example:
        >>> detection: Detection = {
        ...     "bbox": [100.0, 200.0, 300.0, 400.0],
        ...     "conf": 0.95,
        ...     "class_id": 0,
        ...     "class_name": "person"
        ... }
    """

    # Required fields
    bbox: Required[list[float]]
    conf: Required[float]
    class_id: Required[int]

    # Optional fields (added by post-processing)
    class_name: NotRequired[str]


# Type alias for readability
DetectionList = list[Detection]
