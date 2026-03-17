"""
Pipeline Orchestrator with Threading.
"""

from __future__ import annotations

import queue
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
from loguru import logger

from src.core.visualizer import DefaultVisualizer


class PipelineOrchestrator:
    """Threading-based pipeline orchestrator."""

    def __init__(
        self,
        source: str,
        detector_args: dict[str, Any],
        config: Any,
        output_path: str | None = None,
        show_preview: bool = False,
        max_frames: int | None = None,
    ):
        self.source = source
        self.detector_args = detector_args
        self.config = config
        self.output_path = output_path
        self.show_preview = show_preview
        self.max_frames = max_frames

        self.batch_size = detector_args.get("batch_size", 1)
        self.queue_size = config.get("pipeline.queue_size", 64)

        self._stop_event = threading.Event()

    def run(self) -> None:
        """Run pipeline with threading."""
        logger.info("Starting VisionPipeline (Threading Mode)...")

        def signal_handler(_sig: int, _frame: Any) -> None:
            logger.warning("Ctrl+C pressed, stopping...")
            self._stop_event.set()
            _force_stop()

        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            video_info = self._get_video_info()

            logger.info(
                f"Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps"
            )
            logger.info(f"Total frames: {video_info['total_frames']}")

            input_size = self.detector_args.get("input_size")
            if input_size:
                h, w = input_size
                logger.info(f"Input size: {w}x{h} (letterbox)")
            logger.info(f"Backend: {self.detector_args.get('backend')}")
            logger.info(f"Model: {self.detector_args.get('model')}")
            logger.info(f"Batch size: {self.batch_size}")

            fps_source = video_info.get("fps", 30)

            # Threading queues
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
            result_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)

            # Store queues globally for force stop
            global _frame_queue_global, _result_queue_global
            _frame_queue_global = frame_queue
            _result_queue_global = result_queue

            # Create threads
            threads = [
                threading.Thread(
                    target=_reader_thread,
                    args=(self.source, frame_queue, self.max_frames, self._stop_event),
                    name="Reader",
                    daemon=True,
                ),
                threading.Thread(
                    target=_inference_thread,
                    args=(
                        frame_queue,
                        result_queue,
                        self.detector_args,
                        self.batch_size,
                        self._stop_event,
                    ),
                    name="Inference",
                    daemon=True,
                ),
                threading.Thread(
                    target=_viewer_thread,
                    args=(
                        result_queue,
                        self.show_preview,
                        self.output_path,
                        fps_source,
                        self._stop_event,
                    ),
                    name="Viewer",
                    daemon=True,
                ),
            ]

            # Start threads
            logger.info("Starting threads...")
            for t in threads:
                t.start()

            # Wait for completion or Ctrl+C
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.1)
                if self._stop_event.is_set():
                    break

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self._stop_event.set()
            signal.signal(signal.SIGINT, original_handler)
            cv2.destroyAllWindows()
            logger.success("Pipeline stopped.")

    def _get_video_info(self) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.source}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info


# =============================================================================
# GLOBALS FOR FORCE STOP
# =============================================================================

_frame_queue_global: queue.Queue | None = None
_result_queue_global: queue.Queue | None = None


def _force_stop() -> None:
    """Unblock all threads on Ctrl+C."""
    global _frame_queue_global, _result_queue_global

    if _frame_queue_global:
        for _ in range(10):
            try:
                _frame_queue_global.put_nowait(None)
            except queue.Full:
                break

    if _result_queue_global:
        for _ in range(10):
            try:
                _result_queue_global.put_nowait(None)
            except queue.Full:
                break


# =============================================================================
# THREAD FUNCTIONS
# =============================================================================


def _reader_thread(
    source: str,
    frame_queue: queue.Queue,
    max_frames: int | None,
    stop_event: threading.Event,
) -> None:
    """Reader thread - reads frames from video source."""
    logger.info(f"[Reader] Started for {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"[Reader] Cannot open: {source}")
        frame_queue.put(None)
        return

    frame_count = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                if stop_event.is_set():
                    break
                continue

            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            if frame_count % 100 == 0:
                logger.debug(f"[Reader] Read {frame_count} frames")

        frame_queue.put(None)

    except Exception as e:
        logger.error(f"[Reader] Error: {e}")
    finally:
        cap.release()
        logger.success(f"[Reader] Finished: {frame_count} frames")


def _inference_thread(
    frame_queue: queue.Queue,
    result_queue: queue.Queue,
    detector_args: dict,
    batch_size: int,
    stop_event: threading.Event,
) -> None:
    """Inference thread - runs detection on batches of frames."""
    from src.detectors.factory import DetectorFactory

    logger.info("[Inference] Initializing model...")

    try:
        detector = DetectorFactory.create(**detector_args)
        logger.success(f"[Inference] Model loaded: {detector}")
    except Exception as e:
        logger.error(f"[Inference] Failed to load model: {e}")
        result_queue.put(None)
        return

    input_size = detector_args.get("input_size")
    if input_size:
        h, w = input_size
        logger.info(f"[Inference] Letterbox input: {w}x{h}")

    frame_count = 0

    try:
        while not stop_event.is_set():
            batch_frames: list[np.ndarray] = []

            for _ in range(batch_size):
                try:
                    frame = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    if stop_event.is_set():
                        result_queue.put(None)
                        return
                    continue

                if frame is None:
                    if batch_frames:
                        _process_batch(batch_frames, detector, result_queue)
                    result_queue.put(None)
                    logger.success(f"[Inference] Finished: {frame_count} frames")
                    return
                batch_frames.append(frame)

            if not batch_frames:
                continue

            _process_batch(batch_frames, detector, result_queue)
            frame_count += len(batch_frames)

            if frame_count % 30 == 0:
                logger.info(f"[Inference] Processed {frame_count} frames")

    except Exception as e:
        logger.error(f"[Inference] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        result_queue.put(None)


def _process_batch(
    batch_frames: list[np.ndarray],
    detector: Any,
    result_queue: queue.Queue,
) -> None:
    """Process a batch of frames and send results to queue."""
    results = detector.predict_batch(batch_frames)

    for orig_frame, detections in zip(batch_frames, results):
        result_queue.put((orig_frame, detections))


def _viewer_thread(
    result_queue: queue.Queue,
    show_preview: bool,
    output_path: str | None,
    fps_source: float,
    stop_event: threading.Event,
) -> None:
    """Viewer thread - visualizes detections and saves output."""
    import os

    logger.info("[Viewer] Started")

    visualizer = DefaultVisualizer()
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    writer = None

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        while not stop_event.is_set():
            try:
                item = result_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, detections = item

            if writer is None and output_path:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps_source, (w, h))
                logger.info(f"[Viewer] Writer: {w}x{h} @ {fps_source}fps")

            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"[Viewer] FPS: {fps:.2f} | Detections: {len(detections)}")

            annotated = visualizer.draw_all(frame, detections, fps, frame_count)

            if writer:
                writer.write(annotated)

            if show_preview:
                try:
                    cv2.imshow("VisionPipeline", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        stop_event.set()
                        break
                except cv2.error:
                    show_preview = False

    except Exception as e:
        logger.error(f"[Viewer] Error: {e}")
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.success(f"[Viewer] Finished: {frame_count} frames")
