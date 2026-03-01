"""
Backend Registry.

Centralized registry for backend capabilities and metadata queries.
"""

from __future__ import annotations

from typing import Any


class BackendRegistry:
    """Central registry for backend capabilities."""

    @classmethod
    def query_input_size(
        cls,
        backend: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Query backend for input size requirements."""
        backend = backend.lower()

        if backend == "triton":
            return cls._query_triton(
                url=kwargs.get("url", "localhost:8000"),
                model_name=kwargs.get("model_name", "yolo"),
            )

        if backend == "onnx":
            return cls._query_onnx(model_path=kwargs.get("model_path"))

        if backend == "tensorrt":
            return cls._query_tensorrt(engine_path=kwargs.get("engine_path"))

        if backend == "pytorch":
            return cls._query_pytorch()

        return None

    @classmethod
    def _query_triton(
        cls,
        url: str = "localhost:8000",
        model_name: str = "yolo",
    ) -> dict[str, Any]:
        """Query Triton server for model input size."""
        from src.detectors.triton_detector import TritonDetector

        return TritonDetector.query_model_info(url=url, model_name=model_name)

    @classmethod
    def _query_onnx(cls, model_path: str | None = None) -> dict[str, Any] | None:
        """Query ONNX model for input size."""
        if model_path is None:
            return None

        try:
            import onnx
        except ImportError:
            return None

        try:
            model = onnx.load(model_path)
            input_info = model.graph.input[0]

            dims = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    dims.append(-1)
                else:
                    dims.append(dim.dim_value)

            if len(dims) >= 4:
                batch, channels, height, width = dims[:4]

                return {
                    "height": None if height == -1 else height,
                    "width": None if width == -1 else width,
                    "is_dynamic": height == -1 or width == -1,
                    "supports_batch": batch == -1,
                    "batch_size": None if batch == -1 else batch,
                    "shape": dims[:4],
                }

        except Exception:
            pass

        return None

    @classmethod
    def _query_tensorrt(cls, engine_path: str | None = None) -> dict[str, Any] | None:
        """Query TensorRT engine for input size."""
        if engine_path is None:
            return None
        return None

    @classmethod
    def _query_pytorch(cls) -> None:
        """Query PyTorch model for input size. Always returns None."""
        return None

    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """Check if backend dependencies are installed."""
        backend = backend.lower()

        module_map = {
            "pytorch": "torch",
            "onnx": "onnxruntime",
            "tensorrt": "tensorrt",
            "triton": "tritonclient.grpc",
        }

        module_name = module_map.get(backend)
        if module_name is None:
            return False

        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
