"""
Base Detector Abstract Class.

This module defines the abstract interface for all object detectors.
Any detector implementation (YOLO, RT-DETR, etc.) must inherit from
this class and implement the predict() method.

Key concepts:
- Abstract Base Class (ABC): Cannot be instantiated directly
- Template method pattern: Defines algorithm structure, subclasses implement details
- Standardized output format: All detectors return the same detection structure
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.constants import COCO_CLASSES, get_class_color, get_class_name
from src.types import Detection


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    All detector implementations must inherit from this class and implement
    the predict() method. The class provides common functionality:
    - Confidence threshold filtering
    - Class ID filtering
    - Drawing bounding boxes with labels

    Attributes:
        conf_threshold (float): Minimum confidence for detections (0.0-1.0)
        class_ids (Optional[Set[int]]): Classes to detect. None = all classes
    """

    def __init__(self, conf_threshold: float = 0.25, class_ids: list[int] | None = None):
        """
        Initialize the base detector.

        Args:
            conf_threshold: Minimum confidence for detections (0.0-1.0)
                           Default: 0.25 (25% confidence)
            class_ids: List of class IDs to detect.
                      None = detect all 80 COCO classes
                      [0] = detect only persons
                      [0, 2, 5] = detect persons, cars, buses

        Example:
            >>> # Detect everything
            >>> detector = MyDetector(conf_threshold=0.3)

            >>> # Detect only persons
            >>> detector = MyDetector(class_ids=[0])

            >>> # Detect persons and vehicles
            >>> detector = MyDetector(class_ids=[0, 2, 3, 5, 7])
        """
        self.conf_threshold: float = conf_threshold

        # Convert list to set for O(1) lookup
        # None means "all classes"
        self.class_ids: set[int] | None = set(class_ids) if class_ids else None

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame.

        This method must be implemented by all subclasses.
        It should return detections in a standardized format.

        Args:
            frame: Input image in BGR format (OpenCV standard)
                   Shape: (height, width, 3)

        Returns:
            List of Detection TypedDicts:
            {
                'bbox': [x1, y1, x2, y2],  # Absolute pixel coordinates
                'conf': 0.95,              # Confidence score (0.0-1.0)
                'class_id': 0,             # COCO class ID (0-79)
            }

        Note:
            - Bounding box is [x1, y1, x2, y2] (top-left, bottom-right)
            - Subclasses should NOT filter by confidence here
              (handled by filter_detections)
        """
        pass

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames.

        Default implementation loops over frames and calls predict().
        Subclasses can override for optimized batch inference.

        Args:
            frames: List of input images in BGR format
                    Each frame shape: (height, width, 3)

        Returns:
            List of detection lists, one per frame

        Example:
            >>> frames = [frame1, frame2, frame3]
            >>> results = detector.predict_batch(frames)
            >>> # results[0] = detections for frame1
            >>> # results[1] = detections for frame2
            >>> # results[2] = detections for frame3
        """
        return [self.predict(frame) for frame in frames]

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        """
        Filter detections by confidence and class.

        This method applies two filters:
        1. Confidence threshold: Remove detections below conf_threshold
        2. Class filter: Keep only specified classes (if class_ids is set)

        Args:
            detections: Raw list of detections from predict()

        Returns:
            Filtered list of detections with class_name added

        Example:
            >>> raw = [{'bbox': [...], 'conf': 0.5, 'class_id': 0}]
            >>> detector = BaseDetector(conf_threshold=0.3, class_ids=[0])
            >>> filtered = detector.filter_detections(raw)
        """
        filtered: list[Detection] = []

        for det in detections:
            # Check confidence threshold
            if det["conf"] < self.conf_threshold:
                continue

            # Check class filter (if specified)
            if self.class_ids is not None and det["class_id"] not in self.class_ids:
                continue

            # Create new detection with class_name
            detection_with_name: Detection = {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "class_id": det["class_id"],
                "class_name": get_class_name(det["class_id"]),
            }
            filtered.append(detection_with_name)

        return filtered

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Each class gets a unique color for easy visual distinction.

        Args:
            frame: Input image in BGR format
            detections: List of detections (with class_name added)
            thickness: Bounding box line thickness (default: 2)
            font_scale: Font scale for labels (default: 0.6)

        Returns:
            Annotated frame (new array, original is not modified)
        """
        # Create a copy to avoid modifying original
        annotated: np.ndarray = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            conf = det["conf"]
            class_id = det["class_id"]
            class_name = det.get("class_name", get_class_name(class_id))

            # Get color for this class
            color = get_class_color(class_id)

            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Create label text: "class_name confidence"
            # Example: "person 0.95"
            label = f"{class_name} {conf:.2f}"

            # Calculate label background size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Draw label background (filled rectangle)
            # Position: above the bounding box
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,  # Filled
            )

            # Draw label text (black for contrast)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                1,
            )

        return annotated

    def __repr__(self) -> str:
        """String representation for debugging."""
        class_filter = f", classes={self.class_ids}" if self.class_ids else ""
        return f"{self.__class__.__name__}(conf={self.conf_threshold}{class_filter})"

    @property
    def num_classes(self) -> int:
        """Number of classes this detector will return."""
        if self.class_ids is None:
            return len(COCO_CLASSES)
        return len(self.class_ids)
