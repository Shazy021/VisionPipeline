"""
Unified Ring Buffer - Zero-Copy Single Buffer for Pipeline.

One shared memory buffer for the entire pipeline:
- Reader writes frames
- Inference reads frames, writes detections
- Viewer reads frames + detections

Lock-free 3-stage synchronization using separate indices:
- write_idx: Reader's position
- process_idx: Inference's position
- read_idx: Viewer's position

Slot lifecycle:
  EMPTY → FRAME_READY → DET_READY → EMPTY
    ↑                                 ↓
    └─────────── Viewer releases ─────┘

Benefits:
- Zero-copy frame access (numpy view on SHM)
- Single memory allocation (efficient for multicam)
- Lock-free synchronization
"""

from __future__ import annotations

import contextlib
import struct
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from src.types import Detection


class UnifiedRingBuffer:
    """
    Zero-copy unified ring buffer for frames + detections.

    Memory layout per slot:
    [valid: 1][frame: H*W*3][num_dets: 4][dets: MAX_DETS * 28]

    Detection (28 bytes): bbox(16) + conf(4) + class_id(4) + padding(4)
    """

    HEADER_SIZE = 32  # 3 indices + reserved, 8 bytes each
    DETECTION_SIZE = 28

    # Slot states
    EMPTY = 0
    FRAME_READY = 1
    DET_READY = 2
    END = 255

    def __init__(
        self,
        name: str,
        num_slots: int,
        height: int,
        width: int,
        max_detections: int = 300,
        create: bool = False,
    ):
        self.name = name
        self.num_slots = num_slots
        self.height = height
        self.width = width
        self.max_detections = max_detections

        self.frame_size = height * width * 3
        self.dets_size = 4 + max_detections * self.DETECTION_SIZE
        self.slot_size = 1 + self.frame_size + self.dets_size
        self.total_size = self.HEADER_SIZE + num_slots * self.slot_size

        if create:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
            self._buf = np.ndarray((self.total_size,), dtype=np.uint8, buffer=self._shm.buf)
            self._buf[:] = 0
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            self._buf = np.ndarray((self.total_size,), dtype=np.uint8, buffer=self._shm.buf)

    @classmethod
    def create(
        cls,
        name: str,
        num_slots: int,
        height: int,
        width: int,
        max_detections: int = 300,
    ) -> UnifiedRingBuffer:
        """Create new SHM buffer (call in main process)."""
        return cls(name, num_slots, height, width, max_detections, create=True)

    @classmethod
    def connect(
        cls,
        name: str,
        num_slots: int,
        height: int,
        width: int,
        max_detections: int = 300,
    ) -> UnifiedRingBuffer:
        """Connect to existing SHM buffer (call in child processes)."""
        return cls(name, num_slots, height, width, max_detections, create=False)

    # ========================================================================
    # Index Management
    # ========================================================================

    def _slot_offset(self, idx: int) -> int:
        """Get byte offset for slot (modular indexing)."""
        return self.HEADER_SIZE + (idx % self.num_slots) * self.slot_size

    def _get_write_idx(self) -> int:
        return int.from_bytes(bytes(self._buf[0:8]), "little")

    def _set_write_idx(self, idx: int) -> None:
        self._buf[0:8] = list(idx.to_bytes(8, "little"))

    def _get_process_idx(self) -> int:
        return int.from_bytes(bytes(self._buf[8:16]), "little")

    def _set_process_idx(self, idx: int) -> None:
        self._buf[8:16] = list(idx.to_bytes(8, "little"))

    def _get_read_idx(self) -> int:
        return int.from_bytes(bytes(self._buf[16:24]), "little")

    def _set_read_idx(self, idx: int) -> None:
        self._buf[16:24] = list(idx.to_bytes(8, "little"))

    # ========================================================================
    # Reader API: write_frame()
    # ========================================================================

    def write_frame(self, frame: np.ndarray, timeout: float = 1.0) -> tuple[int, bool]:
        """
        Write frame to buffer (Reader process).

        Returns:
            (slot_idx, success) - slot_idx for tracking, success flag
        """
        deadline = time.perf_counter() + timeout

        # Wait for free slot (Reader is ahead of Viewer)
        while True:
            write_idx = self._get_write_idx()
            read_idx = self._get_read_idx()

            if write_idx - read_idx < self.num_slots:
                break

            if time.perf_counter() > deadline:
                return -1, False
            time.sleep(0.001)

        offset = self._slot_offset(write_idx)

        # Mark empty while writing
        self._buf[offset] = self.EMPTY

        # Write frame data
        pos = offset + 1
        self._buf[pos : pos + self.frame_size] = frame.flatten()

        # Reset detections count
        pos += self.frame_size
        self._buf[pos : pos + 4] = [0, 0, 0, 0]

        # Mark frame ready for inference
        self._buf[offset] = self.FRAME_READY

        # Advance write index
        self._set_write_idx(write_idx + 1)

        return write_idx, True

    def write_end(self) -> None:
        """Signal end of stream (Reader process)."""
        write_idx = self._get_write_idx()
        offset = self._slot_offset(write_idx)
        self._buf[offset] = self.END
        self._set_write_idx(write_idx + 1)

    # ========================================================================
    # Inference API: get_frame() + write_detections()
    # ========================================================================

    def get_frame(self, timeout: float = 1.0) -> tuple[np.ndarray | None, int, bool]:
        """
        Get frame for inference (zero-copy view).

        Returns:
            (frame_view, slot_idx, is_end)
            frame_view is a numpy view - DO NOT MODIFY, copy if needed
        """
        deadline = time.perf_counter() + timeout

        # Wait for frame (Inference catches up to Reader)
        while True:
            process_idx = self._get_process_idx()
            write_idx = self._get_write_idx()

            if write_idx > process_idx:
                # MEMORY BARRIER: yield to ensure we see latest writes
                # In multiprocessing, without this we might see updated index
                # but stale slot status (END marker)
                time.sleep(0)
                break

            # Check for END signal even while waiting (reader might have finished)
            offset = self._slot_offset(process_idx)
            if self._buf[offset] == self.END:
                # Advance past END marker
                self._set_process_idx(process_idx + 1)
                return None, -1, True

            if time.perf_counter() > deadline:
                return None, -1, False
            time.sleep(0.001)

        offset = self._slot_offset(process_idx)

        # Check for end signal
        valid = self._buf[offset]
        if valid == self.END:
            self._set_process_idx(process_idx + 1)
            return None, -1, True

        # Wait for FRAME_READY state
        while self._buf[offset] != self.FRAME_READY:
            if time.perf_counter() > deadline:
                return None, -1, False
            time.sleep(0.001)

        # ADVANCE process_idx HERE - slot is now "claimed" by Inference
        self._set_process_idx(process_idx + 1)

        # Return zero-copy view of frame
        pos = offset + 1
        frame_view = self._buf[pos : pos + self.frame_size].reshape((self.height, self.width, 3))

        return frame_view, process_idx, False

    def write_detections(
        self,
        slot_idx: int,
        detections: list[Detection],
    ) -> None:
        """
        Write detections to slot (Inference process).

        Args:
            slot_idx: Slot index from get_frame()
            detections: List of detection dicts
        """
        offset = self._slot_offset(slot_idx)

        # Write detections
        pos = offset + 1 + self.frame_size
        num_dets = min(len(detections), self.max_detections)

        # Write count
        self._buf[pos : pos + 4] = list(struct.pack("<I", num_dets))
        pos += 4

        # Write each detection
        for det in detections[:num_dets]:
            bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
            conf = float(det.get("conf", 0.0))
            class_id = int(det.get("class_id", 0))

            self._buf[pos : pos + 16] = list(struct.pack("<ffff", *bbox))
            self._buf[pos + 16 : pos + 20] = list(struct.pack("<f", conf))
            self._buf[pos + 20 : pos + 24] = list(struct.pack("<i", class_id))
            pos += self.DETECTION_SIZE

        # Mark detections ready
        self._buf[offset] = self.DET_READY

    def write_end_inference(self) -> None:
        """Signal end of stream from Inference process."""
        process_idx = self._get_process_idx()
        offset = self._slot_offset(process_idx)
        self._buf[offset] = self.END
        self._set_process_idx(process_idx + 1)

    # ========================================================================
    # Viewer API: get_result() + release_slot()
    # ========================================================================

    def get_result(self, timeout: float = 1.0) -> tuple[np.ndarray | None, list[Detection], bool]:
        """
        Get frame + detections for display (Viewer process).

        Returns:
            (frame_view, detections, is_end)
            frame_view is a numpy view - COPY before writing to file!
        """
        deadline = time.perf_counter() + timeout

        # Wait for result (Viewer catches up to Inference)
        while True:
            read_idx = self._get_read_idx()
            process_idx = self._get_process_idx()

            if process_idx > read_idx:
                # MEMORY BARRIER: yield to ensure we see latest writes
                time.sleep(0)
                break

            # Check for END signal even while waiting (inference might have finished)
            offset = self._slot_offset(read_idx)
            if self._buf[offset] == self.END:
                # Advance past END marker
                self._set_read_idx(read_idx + 1)
                return None, [], True

            if time.perf_counter() > deadline:
                return None, [], False
            time.sleep(0.001)

        offset = self._slot_offset(read_idx)

        # Check for end signal
        valid = self._buf[offset]
        if valid == self.END:
            self._set_read_idx(read_idx + 1)
            return None, [], True

        # Wait for DET_READY state
        while self._buf[offset] != self.DET_READY:
            if time.perf_counter() > deadline:
                return None, [], False
            time.sleep(0.001)

        # Get zero-copy frame view
        pos = offset + 1
        frame_view = self._buf[pos : pos + self.frame_size].reshape((self.height, self.width, 3))
        pos += self.frame_size

        # Read detections
        num_dets = struct.unpack("<I", bytes(self._buf[pos : pos + 4]))[0]
        pos += 4

        detections: list[Detection] = []
        for _ in range(num_dets):
            bbox = list(struct.unpack("<ffff", bytes(self._buf[pos : pos + 16])))
            conf = struct.unpack("<f", bytes(self._buf[pos + 16 : pos + 20]))[0]
            class_id = struct.unpack("<i", bytes(self._buf[pos + 20 : pos + 24]))[0]
            detections.append({"bbox": bbox, "conf": conf, "class_id": class_id})
            pos += self.DETECTION_SIZE

        # Store slot_idx for release
        self._current_slot = read_idx

        return frame_view, detections, False

    def release_slot(self) -> None:
        """
        Release slot after reading (Viewer process).

        MUST call after processing frame to allow Reader to reuse slot.
        """
        read_idx = self._get_read_idx()
        offset = self._slot_offset(read_idx)

        # Mark slot empty
        self._buf[offset] = self.EMPTY

        # Advance read index
        self._set_read_idx(read_idx + 1)

    # ========================================================================
    # Cleanup
    # ========================================================================

    def close(self) -> None:
        """Close SHM (all processes)."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink SHM (main process only)."""
        with contextlib.suppress(FileNotFoundError):
            self._shm.unlink()

    def __enter__(self) -> UnifiedRingBuffer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
