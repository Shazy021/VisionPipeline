"""
Video Processing Pipeline.

This module implements a multi-process pipeline for video analysis:
    VideoReader -> Inference -> Viewer

Each stage runs in a separate process, communicating via multiprocessing Queues.
This architecture bypasses Python's GIL for true parallelism.

Process Flow:
    1. VideoReader: Reads frames from video file, puts into queue_frames
    2. Inference: Takes frames, runs detection, puts results into queue_results
    3. Viewer: Draws detections, shows preview, saves to output file

Architecture Benefits:
    - Parallel processing (bypasses GIL)
    - Buffered communication (smooths out processing spikes)
    - Easy to scale (can add multiple inference processes)
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
from loguru import logger

from src.constants import get_class_color, get_class_name
from src.types import Detection

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def draw_detection_box(
    frame: np.ndarray, detection: Detection, thickness: int = 2, font_scale: float = 0.6
) -> None:
    """
    Draw a single detection box with label on frame.

    Modifies frame in-place for efficiency.

    Args:
        frame: Image to draw on (BGR format)
        detection: Detection dict with bbox, conf, class_id, class_name
        thickness: Box line thickness
        font_scale: Font scale for label text
    """
    bbox = detection["bbox"]
    conf = detection["conf"]
    class_id = detection["class_id"]
    class_name = detection.get("class_name", get_class_name(class_id))

    # Get color for this class (each class has unique color)
    color = get_class_color(class_id)

    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Create label: "classname 0.95"
    label = f"{class_name} {conf:.2f}"

    # Calculate label size
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    # Handle case where label would go above image
    label_y = y1
    if y1 < label_h + baseline + 5:
        label_y = y1 + label_h + baseline + 5

    # Draw label background (filled rectangle with class color)
    cv2.rectangle(
        frame,
        (x1, label_y - label_h - baseline - 5),
        (x1 + label_w, label_y),
        color,
        -1,  # Filled
    )

    # Draw label text (black for contrast)
    cv2.putText(
        frame,
        label,
        (x1, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),  # Black text
        1,
    )


def draw_legend(frame: np.ndarray, detections: list[Detection], position: str = "right") -> None:
    """
    Draw a legend showing detected classes and their counts.

    Example output:
    +--------------+
    | person: 5    |
    | car: 3       |
    | dog: 1       |
    +--------------+

    Args:
        frame: Image to draw on
        detections: List of detections for counting
        position: "right" or "left" for legend placement
    """
    if not detections:
        return

    # Count detections by class
    class_counts = Counter(det["class_id"] for det in detections)

    # Legend settings
    line_height = 25
    padding = 10
    font_scale = 0.5
    legend_width = 150
    legend_height = len(class_counts) * line_height + padding * 2

    # Calculate position
    h, w = frame.shape[:2]
    x_start = w - legend_width - padding if position == "right" else padding

    y_start = padding + 30  # Below FPS counter

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (30, 30, 30),  # Dark gray
        -1,
    )
    # Blend overlay with original (transparency effect)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw each class count
    y = y_start + padding + 15
    for class_id, count in sorted(class_counts.items()):
        class_name = get_class_name(class_id)
        color = get_class_color(class_id)

        # Draw color indicator (small square)
        cv2.rectangle(frame, (x_start + 5, y - 10), (x_start + 20, y + 5), color, -1)

        # Draw text
        text = f"{class_name}: {count}"
        cv2.putText(
            frame,
            text,
            (x_start + 25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            1,
        )
        y += line_height


def draw_fps(frame: np.ndarray, fps: float, frame_count: int) -> None:
    """
    Draw FPS counter and frame count in top-left corner.

    Args:
        frame: Image to draw on
        fps: Current frames per second
        frame_count: Total frames processed
    """
    # FPS text
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,  # Yellow
    )

    # Frame count
    frame_text = f"Frame: {frame_count}"
    cv2.putText(
        frame,
        frame_text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,  # Yellow
    )


# =============================================================================
# PROCESS FUNCTIONS
# =============================================================================


def video_reader_process(
    source_path: str, queue_frames: mp.Queue, max_frames: int | None = None
) -> None:
    """
    Read video frames and put them into the queue.

    This process reads frames one by one from the video file
    and pushes them to the processing queue.

    Args:
        source_path: Path to input video file
        queue_frames: Queue to send frames to inference process
        max_frames: Maximum frames to read (None = all frames)
    """
    logger.info(f"[Reader] Started for {source_path}")
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        logger.error(f"[Reader] Cannot open video: {source_path}")
        queue_frames.put(None)
        return

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            queue_frames.put(frame)

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.debug(f"[Reader] Read {frame_count} frames")

    except Exception as e:
        logger.error(f"[Reader] Error: {e}")
    finally:
        cap.release()
        queue_frames.put(None)  # Stop signal
        logger.success(f"[Reader] Finished. Total frames: {frame_count}")


def inference_process(
    queue_frames: mp.Queue, queue_results: mp.Queue, detector_args: dict[str, Any]
) -> None:
    """
    Take frames, run inference, put results.

    This process initializes the detector once (important for GPU models)
    and then processes frames as they arrive.

    Args:
        queue_frames: Queue to receive frames from reader
        queue_results: Queue to send (frame, detections) to viewer
        detector_args: Arguments for DetectorFactory.create()
    """
    from src.detectors.factory import DetectorFactory

    logger.info("[Inference] Initializing model...")

    try:
        detector = DetectorFactory.create(**detector_args)
        logger.success(f"[Inference] Model loaded: {detector}")
    except Exception as e:
        logger.error(f"[Inference] Failed to load model: {e}")
        queue_results.put(None)
        return

    frame_count = 0
    try:
        while True:
            frame = queue_frames.get()

            if frame is None:
                queue_results.put(None)
                break

            # Run inference and apply filtering
            detections = detector.predict(frame)
            detections = detector.filter_detections(detections)

            queue_results.put((frame, detections))

            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(f"[Inference] Processed {frame_count} frames")

    except Exception as e:
        logger.error(f"[Inference] Error: {e}")
    finally:
        logger.success(f"[Inference] Finished. Total frames: {frame_count}")


def viewer_process(
    queue_results: mp.Queue, show_preview: bool, output_path: str | None, fps_source: int
) -> None:
    """
    Draw results, show preview, and save to file.

    This process handles all visualization and output:
    - Draws bounding boxes with class labels
    - Shows legend with detected classes
    - Displays FPS counter
    - Writes output video if path specified

    Args:
        queue_results: Queue to receive (frame, detections) from inference
        show_preview: Whether to display preview window
        output_path: Path to save output video (None = no save)
        fps_source: FPS for output video (from source)
    """
    logger.info("[Viewer] Started")

    frame_count = 0
    start_time = time.time()
    fps = 0.0
    writer = None

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        while True:
            item = queue_results.get()

            if item is None:
                break

            frame, detections = item

            # Initialize video writer on first frame
            if writer is None and output_path:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps_source, (w, h))
                logger.info(f"[Viewer] Writer initialized: {w}x{h} @ {fps_source}FPS")

            frame_count += 1

            # Calculate FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"[Viewer] FPS: {fps:.2f} | Detections: {len(detections)}")

            # =================================================================
            # VISUALIZATION
            # =================================================================

            # Draw all detection boxes with labels
            for det in detections:
                draw_detection_box(frame, det)

            # Draw legend (class counts)
            draw_legend(frame, detections)

            # Draw FPS counter
            draw_fps(frame, fps, frame_count)

            # Write to output file
            if writer:
                writer.write(frame)

            # Show preview window
            if show_preview:
                try:
                    cv2.imshow("VisionPipeline", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("[Viewer] User requested quit")
                        break
                except cv2.error as e:
                    logger.warning(f"[Viewer] Display error: {e}")
                    show_preview = False  # Disable further attempts

    except Exception as e:
        logger.error(f"[Viewer] Error: {e}")
    finally:
        if writer:
            writer.release()
            logger.success(f"[Viewer] Video saved: {output_path}")
        cv2.destroyAllWindows()
        logger.success(f"[Viewer] Finished. Total frames: {frame_count}")
