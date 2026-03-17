"""
Multiprocessing Pipeline with Unified Ring Buffer (Zero-Copy).

Key optimizations:
- Single buffer (no frame copying between stages)
- Zero-copy frame access (numpy view on SHM)
- Lock-free 3-stage synchronization
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any

import cv2
import numpy as np
from loguru import logger


class UnifiedPipeline:
    """Multiprocessing pipeline with unified zero-copy ring buffer."""

    MAX_DETECTIONS = 300
    BATCH_TIMEOUT = 0.003  # 30ms timeout for partial batch

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

        # Get video info
        cap = cv2.VideoCapture(source)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Buffer slots = 4x batch size (enough for pipeline depth)
        self.num_slots = max(8, self.batch_size * 4)

        # SHM name
        self.shm_name = f"vp_unified_{os.getpid()}"

    def run(self) -> None:
        """Run pipeline."""
        from src.core.shm_ring_buffer import UnifiedRingBuffer

        logger.info("=" * 60)
        logger.info("VisionPipeline - Unified Zero-Copy Ring Buffer")
        logger.info("=" * 60)
        logger.info(
            f"Video: {self.width}x{self.height} @ {self.fps:.1f}fps, {self.total_frames} frames"
        )
        logger.info(f"Batch: {self.batch_size}, Buffer: {self.num_slots} slots")
        logger.info(f"SHM: {self.shm_name}")
        logger.info("=" * 60)

        # CREATE SHM IN MAIN PROCESS
        buf = UnifiedRingBuffer.create(
            self.shm_name,
            self.num_slots,
            self.height,
            self.width,
            self.MAX_DETECTIONS,
        )
        logger.info("Unified SHM buffer created in main process")

        # Create processes
        processes = [
            mp.Process(
                target=_reader_proc,
                args=(
                    self.source,
                    self.shm_name,
                    self.num_slots,
                    self.height,
                    self.width,
                    self.max_frames,
                ),
            ),
            mp.Process(
                target=_inference_proc,
                args=(
                    self.shm_name,
                    self.num_slots,
                    self.height,
                    self.width,
                    self.detector_args,
                    self.MAX_DETECTIONS,
                    self.batch_size,
                    self.BATCH_TIMEOUT,
                ),
            ),
            mp.Process(
                target=_viewer_proc,
                args=(
                    self.shm_name,
                    self.num_slots,
                    self.height,
                    self.width,
                    self.fps,
                    self.show_preview,
                    self.output_path,
                    self.MAX_DETECTIONS,
                ),
            ),
        ]

        for p in processes:
            p.start()

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()

        # Cleanup SHM
        buf.close()
        buf.unlink()

        logger.success("Pipeline finished.")


def _reader_proc(
    source: str,
    shm_name: str,
    num_slots: int,
    height: int,
    width: int,
    max_frames: int | None,
) -> None:
    """Reader: video -> SHM buffer."""
    from src.core.shm_ring_buffer import UnifiedRingBuffer

    logger.info(f"[Reader] Started: {source}")

    buf = UnifiedRingBuffer.connect(shm_name, num_slots, height, width)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("[Reader] Cannot open video")
        buf.write_end()
        buf.close()
        return

    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            slot_idx, success = buf.write_frame(frame, timeout=0.05)
            if not success:
                continue

            count += 1
            if max_frames and count >= max_frames:
                break
            if count % 100 == 0:
                logger.debug(f"[Reader] {count} frames")

        buf.write_end()
        logger.success(f"[Reader] Done: {count} frames")
    finally:
        cap.release()
        buf.close()


def _inference_proc(
    shm_name: str,
    num_slots: int,
    height: int,
    width: int,
    detector_args: dict,
    max_dets: int,
    batch_size: int,
    batch_timeout: float,
) -> None:
    """Inference: read frames from buffer -> model -> write detections."""
    from src.core.shm_ring_buffer import UnifiedRingBuffer
    from src.detectors.factory import DetectorFactory

    logger.info("[Inference] Connecting to SHM...")

    buf = UnifiedRingBuffer.connect(shm_name, num_slots, height, width, max_dets)
    logger.info("[Inference] SHM connected, loading model...")

    try:
        detector = DetectorFactory.create(**detector_args)
        logger.success("[Inference] Model ready")
    except Exception as e:
        logger.error(f"[Inference] Load failed: {e}")
        buf.write_end_inference()
        buf.close()
        return

    count = 0
    try:
        while True:
            # ADAPTIVE BATCH with zero-copy
            batch_frames: list[np.ndarray] = []
            batch_slots: list[int] = []
            start = time.perf_counter()

            while len(batch_frames) < batch_size:
                frame_view, slot_idx, is_end = buf.get_frame(timeout=0.01)

                if is_end:
                    # Process remaining batch
                    if batch_frames:
                        results = detector.predict_batch(batch_frames)
                        for slot, dets in zip(batch_slots, results):
                            buf.write_detections(slot, dets)
                    buf.write_end_inference()
                    logger.success(f"[Inference] Done: {count} frames")
                    return

                if frame_view is not None:
                    # COPY frame for batch (model may need contiguous array)
                    batch_frames.append(frame_view.copy())
                    batch_slots.append(slot_idx)
                    start = time.perf_counter()
                elif time.perf_counter() - start > batch_timeout and batch_frames:
                    break

            if not batch_frames:
                continue

            # Run inference
            results = detector.predict_batch(batch_frames)

            # Write detections to each slot
            for slot, dets in zip(batch_slots, results):
                buf.write_detections(slot, dets)
                count += 1

            if count % 30 == 0:
                logger.info(f"[Inference] {count} frames (batch={len(batch_frames)})")

    except Exception as e:
        logger.error(f"[Inference] Error: {e}")
        import traceback

        traceback.print_exc()
        buf.write_end_inference()
    finally:
        buf.close()


def _viewer_proc(
    shm_name: str,
    num_slots: int,
    height: int,
    width: int,
    fps_src: float,
    show: bool,
    output: str | None,
    max_dets: int,
) -> None:
    """Viewer: read frame + detections -> display/save."""
    from src.core.shm_ring_buffer import UnifiedRingBuffer
    from src.core.visualizer import DefaultVisualizer

    logger.info("[Viewer] Started")

    buf = UnifiedRingBuffer.connect(shm_name, num_slots, height, width, max_dets)
    viz = DefaultVisualizer()

    count = 0
    start = time.time()
    fps = 0.0
    writer = None

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    try:
        while True:
            frame_view, dets, is_end = buf.get_result(timeout=2.0)

            if is_end:
                break
            if frame_view is None:
                continue

            if writer is None and output:
                writer = cv2.VideoWriter(
                    output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_src,
                    (width, height),
                )

            count += 1

            if count % 30 == 0:
                fps = count / (time.time() - start)
                logger.info(f"[Viewer] FPS: {fps:.1f} | Dets: {len(dets)}")

            # Draw on frame (need copy for visualization)
            annotated = viz.draw_all(frame_view, dets, fps, count)

            if writer:
                # Must copy for writer
                writer.write(annotated)

            if show:
                cv2.imshow("VisionPipeline", annotated)
                if cv2.waitKey(1) in (ord("q"), 27):
                    break

            # RELEASE SLOT - critical for lock-free!
            buf.release_slot()

        logger.success(f"[Viewer] Done: {count} frames")
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        buf.close()
