"""
Model Export Utilities.

This module provides utilities for exporting YOLO detection models to optimized
inference formats: ONNX and TensorRT.

Export formats:
- ONNX: Cross-platform, CPU/GPU compatible, 1.5-2x speedup
- TensorRT: NVIDIA GPU optimized, FP16/INT8 support

Usage:
    # Export to ONNX
    ModelExporter.export_to_onnx("yolo11n.pt", imgsz=640)

    # Export to TensorRT with FP16
    ModelExporter.export_to_tensorrt("yolo11n.pt", fp16=True)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from ultralytics import YOLO

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not installed. Model export unavailable.")


class ModelExporter:
    """
    Export utility for converting YOLO detection models to optimized formats.

    Supports:
    - ONNX: Cross-platform, CPU/GPU compatible
    - TensorRT: NVIDIA GPU optimized (FP16, INT8 with calibration)

    All methods are static for easy access without instantiation.
    """

    @staticmethod
    def _load_model(model_path: str) -> YOLO:
        """
        Load YOLO model.

        Args:
            model_path: Path to model file

        Returns:
            Loaded Ultralytics YOLO model instance
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library required for model export")

        return YOLO(model_path)

    @staticmethod
    def export_to_onnx(
        model_path: str,
        output_dir: str = "weights",  # noqa: ARG004
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = False,
        imgsz: int = 640,
    ) -> str:
        """
        Export PyTorch model to ONNX format.

        ONNX provides:
        - Cross-platform compatibility (Windows/Linux/macOS)
        - CPU and GPU support (CUDA)
        - 1.5-2x speedup vs PyTorch
        - Smaller file size

        Args:
            model_path: Path to PyTorch model (.pt file)
            output_dir: Directory to save exported model (unused, kept for API compat)
            opset: ONNX opset version (default: 17)
            simplify: Simplify ONNX graph (default: True)
            dynamic: Dynamic batch size support (default: False)
            imgsz: Input image size (default: 640)

        Returns:
            Path to exported ONNX model

        Raises:
            FileNotFoundError: If source model doesn't exist
            RuntimeError: If export fails

        Example:
            >>> onnx_path = ModelExporter.export_to_onnx("yolo11n.pt", imgsz=640)
            >>> print(f"Exported to: {onnx_path}")
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        logger.info(f"📦 Exporting to ONNX: {model_path}")
        logger.info(
            f"   Opset: {opset} | Size: {imgsz}x{imgsz} | Simplify: {simplify} | Dynamic: {dynamic}"
        )

        start_time = time.time()

        try:
            model = ModelExporter._load_model(model_path)

            # Export to ONNX
            onnx_path = model.export(
                format="onnx", opset=opset, simplify=simplify, dynamic=dynamic, imgsz=imgsz
            )

            elapsed = time.time() - start_time
            onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)

            logger.success("✅ ONNX export complete!")
            logger.info(f"   Output: {onnx_path}")
            logger.info(f"   Size: {onnx_size:.2f} MB | Time: {elapsed:.1f}s")

            return str(onnx_path)

        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            raise RuntimeError(f"ONNX export failed: {e}") from None

    @staticmethod
    def export_to_tensorrt(
        model_path: str,
        output_dir: str = "weights",  # noqa: ARG004
        fp16: bool = False,
        int8: bool = False,
        workspace: int = 4,
        imgsz: int = 640,
        opset: int = 17,
        dynamic: bool = False,
        batch_size: int = 1,
        data: str | None = None,
    ) -> str:
        """
        Export PyTorch model to TensorRT engine.

        TensorRT provides:
        - Maximum performance on NVIDIA GPUs
        - FP16 and INT8 quantization
        - Layer fusion and kernel auto-tuning
        - 2-5x speedup vs ONNX

        Args:
            model_path: Path to PyTorch model (.pt file)
            output_dir: Directory to save exported engine (unused, kept for API compat)
            fp16: Use FP16 precision (default: False)
            int8: Use INT8 quantization (default: False).
                  Note: INT8 requires calibration data for stable accuracy.
            workspace: GPU workspace size in GB (default: 4)
            imgsz: Input image size (default: 640)
            opset: ONNX opset for intermediate conversion (default: 17)
            data: Path to dataset config (YAML) for INT8 calibration.

        Returns:
            Path to exported TensorRT engine

        Raises:
            FileNotFoundError: If source model doesn't exist
            RuntimeError: If export fails (no GPU, wrong CUDA version)

        Example:
            >>> # FP16 export (recommended for most GPUs)
            >>> trt_path = ModelExporter.export_to_tensorrt("yolo11n.pt", fp16=True)

            >>> # INT8 export (requires calibration data)
            >>> trt_path = ModelExporter.export_to_tensorrt(
            ...     "yolo11n.pt", int8=True, data="coco.yaml"
            ... )
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        # Determine precision
        if int8:
            precision = "INT8"
            logger.warning("⚠️  INT8 mode enabled - calibration data recommended")
            if data is None:
                logger.warning("   No calibration data provided. Accuracy may drop.")
        elif fp16:
            precision = "FP16"
        else:
            precision = "FP32"

        logger.info(f"📦 Exporting to TensorRT: {model_path}")
        logger.info(
            f"   Precision: {precision} | Size: {imgsz}x{imgsz} | Batch: {batch_size} | Dynamic: {dynamic} | Workspace: {workspace}GB"
        )

        start_time = time.time()

        try:
            model = ModelExporter._load_model(model_path)

            # Export arguments
            export_args = {
                "format": "engine",
                "half": fp16,
                "int8": int8,
                "opset": opset,
                "workspace": workspace,
                "dynamic": dynamic,
                "batch": batch_size,
                "imgsz": imgsz,
            }

            # Pass calibration data for INT8
            if data:
                export_args["data"] = data
                logger.info(f"   Calibration data: {data}")

            # Export to TensorRT
            trt_path = model.export(**export_args)

            elapsed = time.time() - start_time
            trt_size = Path(trt_path).stat().st_size / (1024 * 1024)

            logger.success("✅ TensorRT export complete!")
            logger.info(f"   Output: {trt_path}")
            logger.info(f"   Size: {trt_size:.2f} MB | Time: {elapsed:.1f}s")

            return str(trt_path)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ TensorRT export failed after {elapsed:.1f}s: {e}")

            # Helpful error for INT8
            if int8 and ("calibration" in str(e).lower() or "data" in str(e).lower()):
                raise RuntimeError(
                    f"INT8 export failed: {e}\n"
                    "INT8 requires calibration data. "
                    "Provide a YAML dataset config via 'data' parameter."
                ) from None

            raise RuntimeError(f"TensorRT export failed: {e}") from None

    @staticmethod
    def export_model(
        model_path: str,
        format: str,
        output_dir: str = "weights",
        **kwargs: Any,
    ) -> str:
        """
        Universal export method supporting multiple formats.

        Args:
            model_path: Path to source PyTorch model
            format: Export format ('onnx', 'tensorrt', 'engine', 'trt')
            output_dir: Output directory (unused, kept for API compat)
            **kwargs: Format-specific arguments (fp16, int8, imgsz, etc.)

        Returns:
            Path to exported model

        Raises:
            ValueError: If format is not supported

        Example:
            >>> # Export to ONNX
            >>> path = ModelExporter.export_model("yolo11n.pt", "onnx")

            >>> # Export to TensorRT with FP16
            >>> path = ModelExporter.export_model("yolo11n.pt", "tensorrt", fp16=True)
        """
        format_lower = format.lower()

        if format_lower == "onnx":
            return ModelExporter.export_to_onnx(model_path, output_dir, **kwargs)
        elif format_lower in ["tensorrt", "engine", "trt"]:
            return ModelExporter.export_to_tensorrt(model_path, output_dir, **kwargs)
        else:
            raise ValueError(
                f"Unsupported format: '{format}'. "
                f"Supported: 'onnx', 'tensorrt' (or 'engine', 'trt')"
            )
