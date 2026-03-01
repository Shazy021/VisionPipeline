"""
Triton Inference Server Detector.

Uses gRPC for communication (works on Windows and Linux).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger

from src.detectors.base_detector import BaseDetector
from src.types import Detection

# =============================================================================
# Internal gRPC Client
# =============================================================================

_grpc: Any = None


def _get_grpc() -> Any:
    """Lazily import tritonclient.grpc."""
    global _grpc
    if _grpc is None:
        try:
            import tritonclient.grpc as grpc_module

            _grpc = grpc_module
        except ImportError as e:
            raise ImportError(
                "tritonclient[grpc] not installed. Run: pip install tritonclient[grpc]"
            ) from e
    return _grpc


def _to_grpc_url(url: str) -> str:
    """Convert URL to gRPC port 8001."""
    if ":8001" in url:
        return url
    if ":8000" in url:
        return url.replace(":8000", ":8001")
    if ":" not in url:
        return f"{url}:8001"
    return url


class _TritonClient:
    """
    Internal gRPC client for Triton server.

    Handles connection, metadata queries, and inference.
    Not intended for direct use - use TritonDetector instead.
    """

    def __init__(self, url: str, check_live: bool = True):
        self.grpc = _get_grpc()
        self.grpc_url = _to_grpc_url(url)
        self._client = self.grpc.InferenceServerClient(self.grpc_url, verbose=False)

        if check_live and not self._client.is_server_live():
            raise ConnectionError(f"Triton server at {self.grpc_url} is not live")

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Query model metadata from server."""
        metadata = self._client.get_model_metadata(model_name)

        for inp in metadata.inputs:
            if inp.name == "images":
                shape = list(inp.shape)  # [batch, channels, height, width]
                batch, channels, height, width = shape
                is_dynamic = height == -1 or width == -1

                return {
                    "height": None if is_dynamic else height,
                    "width": None if is_dynamic else width,
                    "is_dynamic": is_dynamic,
                    "supports_batch": batch == -1,
                    "batch_size": None if batch == -1 else batch,
                    "shape": shape,
                }

        raise ValueError(f"No 'images' input in model '{model_name}'")

    def infer(self, model_name: str, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference and return output array."""
        inputs = self.grpc.InferInput("images", input_tensor.shape, "FP32")
        inputs.set_data_from_numpy(input_tensor)

        outputs = self.grpc.InferRequestedOutput("output0")

        result = self._client.infer(model_name, [inputs], outputs=[outputs])
        return result.as_numpy("output0")


# =============================================================================
# Triton Detector
# =============================================================================


class TritonDetector(BaseDetector):
    """
    Detector using NVIDIA Triton Inference Server via gRPC.

    Input size must be resolved externally via BackendRegistry before creating detector.
    """

    @classmethod
    def query_model_info(cls, url: str, model_name: str) -> dict[str, Any]:
        """
        Query Triton server for model input metadata.

        Used by BackendRegistry to determine input size requirements.
        Can be called before creating detector instance.
        """
        client = _TritonClient(url, check_live=True)
        info = client.get_model_info(model_name)

        if info["is_dynamic"]:
            logger.info(f"📐 Triton '{model_name}' dynamic shape: {info['shape']}")
        else:
            logger.info(f"📐 Triton '{model_name}' input: {info['width']}x{info['height']}")

        return info

    def __init__(
        self,
        model_name: str,
        url: str = "localhost:8000",
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] | None = None,
        class_ids: list[int] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """
        Initialize Triton detector.

        Args:
            model_name: Model name in Triton Model Repository
            url: Triton server URL
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            input_size: Tuple (height, width). Required.
            class_ids: List of class IDs to detect. None = all classes.
        """
        super().__init__(conf_threshold, class_ids)

        if input_size is None:
            raise ValueError(
                "input_size must be provided. "
                "Use BackendRegistry.query_input_size('triton', ...) before creating detector."
            )

        self.model_name = model_name
        self.input_size = input_size
        self.nms_threshold = nms_threshold

        logger.info(f"[Triton] Connecting to {url}...")
        self.client = _TritonClient(url, check_live=False)

        self._scale_x: float | None = None
        self._scale_y: float | None = None
        self._last_frame_shape: tuple[int, int] | None = None

        logger.success(
            f"[Triton] Connected | Model: {model_name} | Input: {input_size[1]}x{input_size[0]}"
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame: resize, BGR→RGB, HWC→CHW, normalize."""
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,  # normalize
            size=(self.input_size[1], self.input_size[0]),  # resize
            swapRB=True,  # BGR→RGB
            crop=False,
        )
        return blob[0]

    def postprocess(self, output: np.ndarray, orig_shape: tuple[int, int, int]) -> list[Detection]:
        """Convert YOLO output to detections with NMS."""
        h, w = orig_shape[:2]

        if self._last_frame_shape != (h, w):
            self._scale_x = w / self.input_size[1]
            self._scale_y = h / self.input_size[0]
            self._last_frame_shape = (h, w)

        predictions = output[0].T
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)

        mask = confs >= self.conf_threshold
        boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]

        if len(boxes) == 0:
            return []

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5

        xyxy[:, [0, 2]] *= self._scale_x
        xyxy[:, [1, 3]] *= self._scale_y

        keep = cv2.dnn.NMSBoxes(xyxy, confs, self.conf_threshold, self.nms_threshold)

        return [
            {"bbox": xyxy[i].tolist(), "conf": float(confs[i]), "class_id": int(class_ids[i])}
            for i in keep.flatten()
        ]

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference via gRPC."""
        input_tensor = self.preprocess(frame)[np.newaxis, ...]

        try:
            output = self.client.infer(self.model_name, input_tensor)
        except Exception as e:
            logger.error(f"[Triton] Inference failed: {e}")
            return []

        return self.filter_detections(self.postprocess(output, frame.shape))
