"""
ONNX Runtime Detector.

This module provides object detection using ONNX Runtime, supporting both
CPU and CUDA GPU execution. It includes custom implementations of:
- Letterbox preprocessing (aspect-ratio preserving resize)
- Non-Maximum Suppression (NMS) for YOLO models
- Coordinate transformation back to original image space

Supported models:
- YOLO (v8, v11, etc.) with NMS
- RT-DETR (no NMS required, uses learned queries)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from loguru import logger

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not installed. Run: pip install onnxruntime-gpu")

from src.detectors.base_detector import BaseDetector
from src.types import Detection


class ONNXDetector(BaseDetector):
    """
    Universal ONNX Runtime detector for YOLO and RT-DETR models.

    Provides cross-platform inference with CPU and CUDA GPU support.
    Includes built-in NMS (Non-Maximum Suppression) for YOLO models.

    Attributes:
        nms_threshold: IoU threshold for NMS filtering
        model_type: Model architecture ('yolo' or 'rtdetr')
        input_h: Model input height in pixels
        input_w: Model input width in pixels
        device: Active execution device ('GPU (CUDA)' or 'CPU')
    """

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool = True,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        model_type: str = "yolo",
        input_size: tuple[int, int] | None = None,
        class_ids: list[int] | None = None,
    ):
        """
        Initialize ONNX detector.

        Args:
            onnx_path: Path to ONNX model file (.onnx)
            use_gpu: Use CUDA GPU if available (default: True)
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            nms_threshold: IoU threshold for NMS (0.0-1.0, default: 0.45)
            model_type: Model architecture - 'yolo' or 'rtdetr'
            input_size: Custom input size as (height, width) tuple
            class_ids: List of class IDs to detect. None = all classes

        Raises:
            ImportError: If onnxruntime is not installed
        """
        super().__init__(conf_threshold, class_ids)

        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")

        self.nms_threshold = nms_threshold
        self.model_type = model_type.lower()
        ort.preload_dlls()

        # Configure execution providers (GPU first if available)
        providers = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Create inference session
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get model I/O metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape_info = self.session.get_inputs()[0].shape

        # Parse or use custom input size
        if input_size is not None:
            self.input_h, self.input_w = input_size
        else:
            self.input_h, self.input_w = self._parse_input_shape(self.input_shape_info)

        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]
        self.device = (
            "GPU (CUDA)" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"
        )

        # Log initialization
        logging.info(f"Loaded ONNX model: {onnx_path}")
        logging.info(f"   Model type: {self.model_type.upper()}")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Input size: {self.input_w}x{self.input_h}")

    def _parse_input_shape(self, shape_info: list[Any]) -> tuple[int, int]:
        """Parse input shape from ONNX model metadata."""
        try:
            if len(shape_info) == 4:
                h, w = shape_info[2], shape_info[3]
                if isinstance(h, str) or isinstance(w, str):
                    return 640, 640
                return int(h), int(w)
        except (TypeError, IndexError):
            pass
        return 640, 640

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocess frame with letterbox resize.

        Performs aspect-ratio-preserving resize with padding (letterbox)
        and normalizes pixel values to [0, 1] range.

        Args:
            frame: Input BGR image from OpenCV

        Returns:
            Tuple of (input_tensor, scale, padding)
        """
        img_h, img_w = frame.shape[:2]

        # Calculate scale to fit image in target size
        scale = min(self.input_w / img_w, self.input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        pad_w = (self.input_w - new_w) // 2
        pad_h = (self.input_h - new_h) // 2

        # Add padding using copyMakeBorder (faster than manual array assignment)
        padded = cv2.copyMakeBorder(
            resized,
            pad_h,  # top
            self.input_h - new_h - pad_h,  # bottom
            pad_w,  # left
            self.input_w - new_w - pad_w,  # right
            cv2.BORDER_CONSTANT,
            value=114,
        )

        # Create blob: normalizes to [0,1], transposes HWC->CHW, adds batch dimension
        # Note: swapRB=False because we want to keep BGR order (as OpenCV loads)
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(self.input_w, self.input_h),
            mean=0,
            swapRB=False,
            crop=False,
        )

        # blob shape is (1, 3, H, W) – exactly what ONNX Runtime expects
        return blob, scale, (pad_w, pad_h)

    def non_max_suppression(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray | None = None
    ) -> list[int]:
        """
        Perform Non-Maximum Suppression on bounding boxes.

        Supports class-aware NMS (boxes of different classes don't suppress each other).

        Args:
            boxes: Bounding boxes in xyxy format [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs for each box [N] (optional, for class-aware NMS)

        Returns:
            List of indices to keep after NMS
        """
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            # Class-aware NMS: only suppress boxes of same class
            if class_ids is not None:
                same_class = class_ids[order[1:]] == class_ids[i]
                iou = np.where(same_class, iou, 0.0)

            # Keep boxes with IoU <= threshold
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def postprocess_yolo(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
        scale: float,
        padding: tuple[int, int],
    ) -> list[Detection]:
        """
        Postprocess YOLO model outputs.

        Handles YOLOv8 output format [1, 84, N] or [1, N, 84],
        applies confidence filtering and NMS.

        Args:
            outputs: Raw ONNX outputs
            original_shape: Original image (height, width)
            scale: Preprocessing scale factor
            padding: Preprocessing padding (pad_w, pad_h)

        Returns:
            List of Detection dicts
        """
        output = outputs[0]

        # Remove batch dimension
        if len(output.shape) == 3:
            output = output[0]

        # Handle [84, N] format -> transpose to [N, 84]
        if output.shape[0] == 84:
            output = output.T

        # Split output into boxes and class scores
        boxes = output[:, :4]  # [N, 4] - cx, cy, w, h
        class_scores = output[:, 4:]  # [N, 80] - class probabilities

        # Get best class for each detection
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        mask = scores >= self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert center format (xywh) to corner format (xyxy)
        cx, cy, w, h = filtered_boxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply class-aware NMS
        keep_indices = self.non_max_suppression(boxes_xyxy, filtered_scores, filtered_class_ids)

        return self._boxes_to_detections(
            boxes_xyxy[keep_indices],
            filtered_scores[keep_indices],
            filtered_class_ids[keep_indices],
            original_shape,
            scale,
            padding,
        )

    def postprocess_rtdetr(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
        scale: float,
        padding: tuple[int, int],
    ) -> list[Detection]:
        """
        Postprocess RT-DETR model outputs.

        RT-DETR outputs normalized coordinates in [1, 300, 84] format.
        No NMS required as transformer uses set prediction.

        Args:
            outputs: Raw ONNX outputs
            original_shape: Original image (height, width)
            scale: Preprocessing scale factor
            padding: Preprocessing padding

        Returns:
            List of Detection dicts
        """
        output = outputs[0]

        # Remove batch dimension
        if len(output.shape) == 3:
            output = output[0]  # [300, 84]

        # Split output into boxes and class scores
        boxes = output[:, :4]  # [300, 4] - normalized cxcywh
        class_scores = output[:, 4:]  # [300, 80] - class probabilities

        # Get best class for each detection
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = scores >= self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert normalized cxcywh to xyxy in pixel coordinates
        cx, cy, w, h = filtered_boxes.T

        x1 = (cx - w / 2) * self.input_w
        y1 = (cy - h / 2) * self.input_h
        x2 = (cx + w / 2) * self.input_w
        y2 = (cy + h / 2) * self.input_h

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        return self._boxes_to_detections(
            boxes_xyxy, filtered_scores, filtered_class_ids, original_shape, scale, padding
        )

    def _boxes_to_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        padding: tuple[int, int],
    ) -> list[Detection]:
        """
        Convert boxes to final detection format.

        Maps boxes from model input space back to original image
        coordinates by removing padding and scaling.

        Args:
            boxes: Bounding boxes in xyxy format [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            original_shape: Original image (height, width)
            scale: Preprocessing scale factor
            padding: Preprocessing padding (pad_w, pad_h)

        Returns:
            List of Detection dicts
        """
        pad_w, pad_h = padding
        img_h, img_w = original_shape

        detections: list[Detection] = []
        for box, conf, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            # Remove padding and scale back to original coordinates
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            # Clip to image boundaries
            x1 = np.clip(x1, 0, img_w)
            y1 = np.clip(y1, 0, img_h)
            x2 = np.clip(x2, 0, img_w)
            y2 = np.clip(y2, 0, img_h)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detection: Detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "class_id": int(cls_id),
            }
            detections.append(detection)

        return detections

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run ONNX inference on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of Detection dicts
        """
        original_shape = frame.shape[:2]

        t0 = time.perf_counter()

        # Preprocess frame
        input_tensor, scale, padding = self.preprocess(frame)
        t1 = time.perf_counter()

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        t2 = time.perf_counter()

        # Postprocess based on model type
        if self.model_type == "yolo":
            detections = self.postprocess_yolo(outputs, original_shape, scale, padding)
        else:
            detections = self.postprocess_rtdetr(outputs, original_shape, scale, padding)

        t3 = time.perf_counter()

        logger.debug(
            f"pre={1000 * (t1 - t0):.1f}ms | "
            f"infer={1000 * (t2 - t1):.1f}ms | "
            f"post={1000 * (t3 - t2):.1f}ms"
        )

        # Apply class filtering and add class names
        return self.filter_detections(detections)

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run ONNX inference on a batch of frames.

        More efficient than individual predict() calls because:
        - Single session.run() call instead of N calls
        - Better GPU utilization

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists, one per frame
        """
        if not frames:
            return []

        # Single frame fallback
        if len(frames) == 1:
            return [self.predict(frames[0])]

        # Preprocess all frames
        batch_tensors = []
        scales = []
        paddings = []

        for frame in frames:
            tensor, scale, padding = self.preprocess(frame)
            batch_tensors.append(tensor[0])  # Remove batch dim for stacking
            scales.append(scale)
            paddings.append(padding)

        # Stack into batch: [N, 3, H, W]
        batch = np.stack(batch_tensors, axis=0)

        # Run batch inference
        outputs = self.session.run(self.output_names, {self.input_name: batch})

        # Postprocess each frame
        results = []
        output = outputs[0]  # [N, 8400, 84] or [N, 84, 8400]

        for i, frame in enumerate(frames):
            original_shape = frame.shape[:2]

            # Extract single frame output
            if self.model_type == "yolo":
                detections = self._postprocess_yolo_single(
                    output[i], original_shape, scales[i], paddings[i]
                )
            else:
                detections = self._postprocess_rtdetr_single(
                    output[i], original_shape, scales[i], paddings[i]
                )

            results.append(self.filter_detections(detections))

        return results

    def _postprocess_yolo_single(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        padding: tuple[int, int],
    ) -> list[Detection]:
        """Postprocess single frame output from YOLO batch."""
        # Handle [84, N] format -> transpose to [N, 84]
        if output.shape[0] == 84:
            output = output.T

        boxes = output[:, :4]
        class_scores = output[:, 4:]

        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = scores >= self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        cx, cy, w, h = filtered_boxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep_indices = self.non_max_suppression(boxes_xyxy, filtered_scores, filtered_class_ids)

        return self._boxes_to_detections(
            boxes_xyxy[keep_indices],
            filtered_scores[keep_indices],
            filtered_class_ids[keep_indices],
            original_shape,
            scale,
            padding,
        )

    def _postprocess_rtdetr_single(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        padding: tuple[int, int],
    ) -> list[Detection]:
        """Postprocess single frame output from RT-DETR batch."""
        boxes = output[:, :4]
        class_scores = output[:, 4:]

        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = scores >= self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        cx, cy, w, h = filtered_boxes.T
        x1 = (cx - w / 2) * self.input_w
        y1 = (cy - h / 2) * self.input_h
        x2 = (cx + w / 2) * self.input_w
        y2 = (cy + h / 2) * self.input_h

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        return self._boxes_to_detections(
            boxes_xyxy,
            filtered_scores,
            filtered_class_ids,
            original_shape,
            scale,
            padding,
        )
