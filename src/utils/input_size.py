"""
Input Size Resolution Module.

Centralized logic for determining model input dimensions across different backends.

Priority: CLI > Backend-specific > Config (auto/fixed)
"""

from __future__ import annotations

import os

from loguru import logger

from src.types import ConfigProtocol, InputSizeResult, VideoInfo

from .utils import get_optimal_size


class InputSizeResolver:
    """
    Centralized resolver for model input dimensions.

    Uses BackendRegistry to query backend-specific requirements,
    keeping this module decoupled from detector implementations.
    """

    def __init__(self, config: ConfigProtocol, video_info: VideoInfo):
        self.config = config
        self.video_info = video_info

    def resolve(
        self,
        backend: str,
        cli_override: tuple[int, int] | None = None,
        model: str = "yolo",
        weights_path: str | None = None,
    ) -> InputSizeResult:
        """
        Resolve input size based on priority chain.

        Priority:
            1. CLI override (--input-size)
            2. Backend-specific (via BackendRegistry)
            3. Config mode: auto (from video) or fixed
        """
        backend = backend.lower()

        # 1. CLI override always wins
        if cli_override is not None:
            h, w = get_optimal_size(cli_override[1], cli_override[0])
            logger.info(f"📐 Input size: {w}x{h} (CLI override)")
            return InputSizeResult(height=h, width=w, source="cli")

        # 2. Backend-specific resolution via Registry
        backend_result = self._resolve_backend(backend, model, weights_path)
        if backend_result is not None:
            return backend_result

        # 3. Config-based resolution
        return self._resolve_from_config()

    def _resolve_backend(
        self,
        backend: str,
        model: str,
        weights_path: str | None,
    ) -> InputSizeResult | None:
        """Resolve input size from backend via Registry."""
        from src.detectors.registry import BackendRegistry

        backend_url = self._get_backend_url(backend)

        info = BackendRegistry.query_input_size(
            backend,
            url=backend_url,
            model_name=model,
            model_path=weights_path,
        )

        if info is None:
            return None

        if not info["is_dynamic"]:
            h, w = info["height"], info["width"]
            logger.info(f"📐 Input size: {w}x{h} (from {backend})")
            return InputSizeResult(
                height=h,
                width=w,
                source=backend,
                backend_url=backend_url,
            )

        # Dynamic size - use config-based resolution
        logger.info(f"📐 {backend} model has dynamic input shape")
        config_result = self._resolve_from_config()
        return InputSizeResult(
            height=config_result.height,
            width=config_result.width,
            source=config_result.source,
            backend_url=backend_url,
        )

    def _get_backend_url(self, backend: str) -> str | None:
        """Get server URL for remote backends."""
        if backend == "triton":
            return os.getenv(
                "TRITON_URL",
                self.config.get("triton.url", "localhost:8000"),
            )
        return None

    def _resolve_from_config(self) -> InputSizeResult:
        """Resolve input size from configuration."""
        mode = self.config.get("inference.input_size.mode", "auto")

        if mode == "auto":
            h = self.video_info.optimal_height
            w = self.video_info.optimal_width
            orig_w = self.video_info.original_width
            orig_h = self.video_info.original_height

            logger.info(f"📐 Input size: {w}x{h} (auto from video {orig_w}x{orig_h})")
            return InputSizeResult(height=h, width=w, source="auto")

        # Fixed mode
        fixed = self.config.get("inference.input_size.fixed_size", [640, 640])
        h, w = get_optimal_size(fixed[1], fixed[0])
        logger.info(f"📐 Input size: {w}x{h} (fixed)")
        return InputSizeResult(height=h, width=w, source="fixed")
