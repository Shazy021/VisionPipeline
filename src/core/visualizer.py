"""
Visualizer Module.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.types import Detection


# =============================================================================
# COLOR CONSTANTS (COCO classes)
# =============================================================================

COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

NUM_CLASSES = len(COCO_CLASSES)


def _generate_class_colors(num_classes: int = NUM_CLASSES) -> list[tuple[int, int, int]]:
    """Generate distinct colors for each class using HSV color space."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


COCO_COLORS: list[tuple[int, int, int]] = _generate_class_colors()


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """Get color for class (BGR format)."""
    if 0 <= class_id < len(COCO_COLORS):
        return COCO_COLORS[class_id]
    np.random.seed(class_id * 17)
    return tuple(np.random.randint(0, 255, 3).tolist())


def get_class_name(class_id: int) -> str:
    """Get class name by ID."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


# =============================================================================
# VISUALIZER
# =============================================================================


class DefaultVisualizer:
    """Default detection visualizer."""

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        show_legend: bool = True,
        show_fps: bool = True,
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.show_legend = show_legend
        self.show_fps = show_fps

    def draw_detection_box(self, frame: np.ndarray, detection: Detection) -> None:
        """Draw single bounding box with label."""
        bbox = detection["bbox"]
        conf = detection["conf"]
        class_id = detection["class_id"]
        class_name = detection.get("class_name", get_class_name(class_id))

        color = get_class_color(class_id)
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )

        label_y = y1
        if y1 < label_h + baseline + 5:
            label_y = y1 + label_h + baseline + 5

        cv2.rectangle(
            frame,
            (x1, label_y - label_h - baseline - 5),
            (x1 + label_w, label_y),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (0, 0, 0),
            1,
        )

    def draw_legend(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        position: str = "right",
    ) -> None:
        """Draw legend with class counts."""
        if not detections:
            return

        class_counts = Counter(det["class_id"] for det in detections)

        line_height = 25
        padding = 10
        legend_width = 150
        legend_height = len(class_counts) * line_height + padding * 2

        h, w = frame.shape[:2]
        x_start = w - legend_width - padding if position == "right" else padding
        y_start = padding + 30

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + legend_width, y_start + legend_height),
            (30, 30, 30),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = y_start + padding + 15
        for class_id, count in sorted(class_counts.items()):
            class_name = get_class_name(class_id)
            color = get_class_color(class_id)

            cv2.rectangle(frame, (x_start + 5, y - 10), (x_start + 20, y + 5), color, -1)

            text = f"{class_name}: {count}"
            cv2.putText(
                frame,
                text,
                (x_start + 25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += line_height

    def draw_fps(self, frame: np.ndarray, fps: float, frame_count: int) -> None:
        """Draw FPS and frame counter."""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw all detections on frame."""
        result = frame.copy()
        for det in detections:
            self.draw_detection_box(result, det)
        return result

    def draw_info(
        self,
        frame: np.ndarray,
        fps: float,
        frame_count: int,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw information panel."""
        result = frame.copy()

        if self.show_fps:
            self.draw_fps(result, fps, frame_count)

        if self.show_legend:
            self.draw_legend(result, detections)

        return result

    def draw_all(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        fps: float,
        frame_count: int,
    ) -> np.ndarray:
        """
        Draw everything: detections, legend, FPS.

        Args:
            frame: Source frame
            detections: List of detections
            fps: Current FPS
            frame_count: Frame number

        Returns:
            Annotated frame
        """
        result = frame.copy()

        # Draw all bounding boxes
        for det in detections:
            self.draw_detection_box(result, det)

        # Draw legend
        if self.show_legend:
            self.draw_legend(result, detections)

        # Draw FPS
        if self.show_fps:
            self.draw_fps(result, fps, frame_count)

        return result


# Alias for convenience
Visualizer = DefaultVisualizer

__all__ = [
    "DefaultVisualizer",
    "Visualizer",
    "get_class_color",
    "get_class_name",
    "COCO_CLASSES",
    "COCO_COLORS",
]
