"""
Triton Inference Server Detector.

This module provides object detection via NVIDIA Triton Inference Server.
It uses HTTP/gRPC communication for remote inference, allowing deployment
on dedicated GPU servers.

Key features:
- Remote inference via HTTP API
- Supports YOLOv8/YOLOv11 models
- Automatic server health checking
- OpenCV NMS for post-processing
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger

from src.detectors.base_detector import BaseDetector
from src.types import Detection

# Lazy import - tritonclient is only loaded when actually needed
TRITON_AVAILABLE = False
httpclient: Any = None


def _ensure_triton() -> None:
    """Lazily import tritonclient when needed."""
    global TRITON_AVAILABLE, httpclient
    if httpclient is not None:
        return
    try:
        import tritonclient.http as _httpclient

        httpclient = _httpclient
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False
        logger.warning("tritonclient not installed. Run: pip install tritonclient[http]")


class TritonDetector(BaseDetector):
    """
    Detector implementation using NVIDIA Triton Inference Server.

    Handles HTTP communication and post-processing of YOLO outputs.
    The actual inference runs on a remote Triton server.

    Attributes:
        model_name: Name of the model in Triton Model Repository
        url: Triton server URL (default: localhost:8000)
        nms_threshold: IoU threshold for Non-Maximum Suppression
        input_size: Model input size (height, width)
        client: Triton HTTP client instance
    """

    def __init__(
        self,
        model_name: str,
        url: str = "localhost:8000",
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        class_ids: list[int] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """
        Initialize Triton detector.

        Args:
            model_name: Name of the model in Triton Model Repository.
            url: Triton server URL (default: localhost:8000).
            conf_threshold: Confidence threshold for filtering (0.0-1.0).
            nms_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0).
            input_size: Tuple (height, width) for model input.
            class_ids: List of class IDs to detect. None = all classes.

        Raises:
            ImportError: If tritonclient is not installed.
            ConnectionError: If Triton server is not reachable.
        """
        super().__init__(conf_threshold, class_ids)

        # Lazy load tritonclient
        _ensure_triton()

        if not TRITON_AVAILABLE:
            raise ImportError(
                "tritonclient not installed or missing http support. "
                "Run: pip install tritonclient[http]"
            )

        self.model_name = model_name
        self.url = url
        self.input_size = input_size
        self.nms_threshold = nms_threshold

        # Initialize Triton client
        try:
            self.client = httpclient.InferenceServerClient(url=url)

            # Check server health
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {url} is not live")

            logger.success(f"Connected to Triton Server: {url} | Model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            raise

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for inference.

        Performs:
        1. Resize to model input size
        2. BGR to RGB conversion
        3. HWC to CHW transpose
        4. Normalize to [0, 1]

        Args:
            frame: Input BGR image from OpenCV

        Returns:
            Preprocessed tensor [3, H, W] as float32
        """
        # Resize to model input size
        img = cv2.resize(frame, self.input_size)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        # Contiguous array and normalize
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        return img

    def postprocess(
        self, output_data: np.ndarray, original_shape: tuple[int, int, int]
    ) -> list[Detection]:
        """
        Post-process Triton model outputs.

        Handles YOLO output format [1, 84, N] or [84, N].
        Applies confidence filtering and NMS.

        Args:
            output_data: Raw model output from Triton
            original_shape: Original frame shape (H, W, C)

        Returns:
            List of Detection dicts with bbox, conf, class_id
        """
        output = output_data[0]  # Shape: (84, 8400) for YOLOv8

        h_orig, w_orig = original_shape[:2]

        # =====================================================================
        # Vectorized extraction
        # =====================================================================
        boxes_xywh = output[:4, :].T  # (N, 4) - cx, cy, w, h
        class_scores = output[4:, :].T  # (N, 80) - class probabilities

        # Get best class for each detection
        scores = class_scores.max(axis=1)
        class_ids_arr = class_scores.argmax(axis=1)

        # Filter by confidence threshold
        mask = scores >= self.conf_threshold

        filtered_boxes = boxes_xywh[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids_arr[mask]

        if len(filtered_boxes) == 0:
            return []

        # =====================================================================
        # Coordinate conversion: xywh -> xyxy
        # =====================================================================
        cx, cy, w, h = (
            filtered_boxes[:, 0],
            filtered_boxes[:, 1],
            filtered_boxes[:, 2],
            filtered_boxes[:, 3],
        )

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Scale to original image size
        scale_x = w_orig / self.input_size[0]
        scale_y = h_orig / self.input_size[1]
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        # =====================================================================
        # Non-Maximum Suppression (using OpenCV)
        # =====================================================================
        boxes_list = boxes_xyxy.astype(int).tolist()
        scores_list = filtered_scores.tolist()

        keep_indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_list,
            scores=scores_list,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )

        # =====================================================================
        # Build detection list
        # =====================================================================
        detections: list[Detection] = []

        if len(keep_indices) > 0:
            # Convert to flat numpy array for iteration
            indices = np.array(keep_indices).flatten()
            for idx in indices:
                detection: Detection = {
                    "bbox": [float(x) for x in boxes_list[idx]],
                    "conf": float(scores_list[idx]),
                    "class_id": int(filtered_class_ids[idx]),
                }
                detections.append(detection)

        return detections

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single frame via Triton HTTP API.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of Detection dicts (filtered and with class_names)
        """
        # Preprocess
        input_tensor = self.preprocess(frame)
        input_batch = input_tensor[np.newaxis, ...]  # Add batch dimension

        # Create Triton input
        inputs = httpclient.InferInput("images", input_batch.shape, "FP32")
        inputs.set_data_from_numpy(input_batch)

        # Request output
        outputs = httpclient.InferRequestedOutput("output0")

        # Run inference
        try:
            results = self.client.infer(
                model_name=self.model_name, inputs=[inputs], outputs=[outputs]
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

        # Get output as numpy array
        output_data = results.as_numpy("output0")

        # Post-process
        detections = self.postprocess(output_data, frame.shape)

        # Apply class filtering and add class names
        return self.filter_detections(detections)
