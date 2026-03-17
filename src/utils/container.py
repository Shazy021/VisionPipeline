"""
DI Container for Unified Pipeline (Zero-Copy).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import cv2
from loguru import logger

if TYPE_CHECKING:
    from src.core.orchestrator import UnifiedPipeline

T = TypeVar("T")
Factory = Callable[[], Any]

STRIDE = 32


class Container:
    """Dependency injection container."""

    def __init__(self) -> None:
        self._config: Any = None
        self._cli_args: Any = None

    @property
    def config(self) -> Any:
        if self._config is None:
            raise RuntimeError("Config not registered")
        return self._config

    @property
    def args(self) -> Any:
        if self._cli_args is None:
            raise RuntimeError("CLI args not registered")
        return self._cli_args

    def register_config(self, config_path: str) -> None:
        from src.utils.config_loader import Config

        if self._config is None:
            self._config = Config(config_path)

    def register_cli_args(self, args: Any) -> None:
        self._cli_args = args

    def get_pipeline(self) -> UnifiedPipeline:
        """Get unified pipeline."""
        from src.core.orchestrator import UnifiedPipeline

        detector_args = self.build_detector_args()

        return UnifiedPipeline(
            source=self.args.source,
            detector_args=detector_args,
            config=self.config,
            output_path=self.args.output,
            show_preview=self.args.show,
            max_frames=self.args.max_frames,
        )

    def build_detector_args(self) -> dict[str, Any]:
        """Build detector arguments for DetectorFactory."""
        config = self.config
        args = self.args

        model = args.model or config.get("models.default_model", "yolo")
        backend = args.backend or config.get("inference.default_backend", "pytorch")
        weights_path = self._resolve_weights_path(model, backend)

        input_size = self._resolve_input_size()

        detector_args: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "weights_path": weights_path,
            "conf_threshold": args.conf or config.get("inference.conf_threshold", 0.25),
            "nms_threshold": args.nms or config.get("inference.nms_threshold", 0.45),
            "input_size": input_size,
            "use_gpu": True,
            "class_ids": config.get("inference.class_ids", None),
            "half": True,
            "augment": False,
            "batch_size": config.get("inference.batch_size", 1),
        }

        if backend == "triton":
            detector_args["triton_url"] = config.get("triton.url", "localhost:8001")

        return detector_args

    def _resolve_weights_path(self, model: str, backend: str) -> str:
        """Get model weights path."""
        args = self.args
        config = self.config

        if backend == "triton":
            return "triton_server_model"

        if args.weights:
            return str(args.weights)

        return str(config.get_model_path(model, backend))

    def _resolve_input_size(self) -> tuple[int, int]:
        """Determine model input size with letterbox (stride 32 aligned)."""
        args = self.args
        config = self.config

        if hasattr(args, "input_size") and args.input_size:
            h, w = args.input_size
            h = self._round_to_stride(h)
            w = self._round_to_stride(w)
            logger.info(f"Input size from CLI: {w}x{h} (letterbox)")
            return (h, w)

        mode = config.get("inference.input_size.mode", "auto")

        if mode == "fixed":
            fixed = config.get("inference.input_size.fixed_size", [640, 640])
            w, h = fixed[0], fixed[1]
            h = self._round_to_stride(h)
            w = self._round_to_stride(w)
            logger.info(f"Input size from config (fixed): {w}x{h}")
            return (h, w)

        if mode == "auto":
            video_size = self._get_video_letterbox_size()
            if video_size:
                h, w = video_size
                logger.info(f"Input size from video (auto letterbox): {w}x{h}")
                return (h, w)

        logger.info("Input size: default 640x640")
        return (640, 640)

    def _get_video_letterbox_size(self) -> tuple[int, int] | None:
        """Calculate letterbox input size from video resolution."""
        args = self.args
        config = self.config

        source = args.source
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.warning(f"Cannot open video for auto size: {source}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        max_size = config.get("inference.input_size.max_size", None)

        if max_size is None:
            new_width = self._round_to_stride(width)
            new_height = self._round_to_stride(height)
        else:
            current_max = max(width, height)

            if current_max <= max_size:
                new_width = self._round_to_stride(width)
                new_height = self._round_to_stride(height)
            else:
                scale = max_size / current_max
                new_width = self._round_to_stride(int(width * scale))
                new_height = self._round_to_stride(int(height * scale))

        return (new_height, new_width)

    @staticmethod
    def _round_to_stride(val: int, stride: int = STRIDE) -> int:
        """Round value up to nearest stride multiple."""
        return max(stride, ((val + stride - 1) // stride) * stride)
