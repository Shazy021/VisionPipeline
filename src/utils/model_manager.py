from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .config_loader import Config

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from src.export.model_exporter import ModelExporter


class ModelManager:
    """Handles model file existence checks and preparation."""

    def __init__(self, config: Config):
        self.auto_download = config.get("automation.auto_download", False)
        self.auto_export = config.get("automation.auto_export", False)
        self.config = config

    def ensure_model(self, weights_path: str, backend: str) -> bool:
        """Ensure model file exists. Download/Export if necessary."""
        path = Path(weights_path)

        if path.exists():
            return True

        logger.warning(f"Model not found: {weights_path}")

        if backend == "pytorch":
            if not self.auto_download:
                raise FileNotFoundError(f"Model not found: {weights_path}")
            return self._download_pt_model(path)

        elif backend in ["onnx", "tensorrt"]:
            pt_path = path.with_suffix(".pt")
            if pt_path.exists():
                if not self.auto_export:
                    raise FileNotFoundError(
                        f"Target model not found: {weights_path}. Source PT found. Use --auto-export."
                    )
                return self._export_model(pt_path, path, backend)

            if self.auto_download and self.auto_export:
                logger.info("Downloading source PT model to export...")
                if self._download_pt_model(pt_path):
                    return self._export_model(pt_path, path, backend)

            raise FileNotFoundError(f"Required model missing: {weights_path}")

        return False

    def _download_pt_model(self, target_path: Path) -> bool:
        """Download PyTorch model using Ultralytics API."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library is required to download models.")

        try:
            logger.info(f"Downloading {target_path.name}...")
            model_name = target_path.stem

            # Instantiation triggers download - YOLO only
            YOLO(model_name)

            downloaded = Path(model_name + ".pt")
            if downloaded.exists() and str(downloaded) != str(target_path):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded.rename(target_path)

            logger.success(f"Model saved to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def _export_model(self, source_pt: Path, target_path: Path, backend: str) -> bool:
        """Export model to ONNX or TensorRT."""
        try:
            logger.info(f"Exporting to {backend.upper()}...")

            export_params = self.config.get(f"export.{backend}", {})
            imgsz = self.config.get("inference.input_size.fixed_size", [640, 640])[0]
            batch_size = self.config.get("inference.batch_size", 1)
            dynamic = bool(export_params.get("dynamic", False))

            if backend == "onnx":
                exported_path = ModelExporter.export_to_onnx(
                    model_path=str(source_pt),
                    opset=int(export_params.get("opset", 17)),
                    simplify=bool(export_params.get("simplify", True)),
                    dynamic=dynamic,
                    imgsz=int(imgsz),
                )
            else:  # tensorrt
                exported_path = ModelExporter.export_to_tensorrt(
                    model_path=str(source_pt),
                    fp16=bool(export_params.get("fp16", False)),
                    int8=bool(export_params.get("int8", False)),
                    workspace=int(export_params.get("workspace_gb", 4)),
                    imgsz=int(imgsz),
                    dynamic=dynamic,
                    batch_size=batch_size,
                )

            exported_path_obj = Path(exported_path)
            if exported_path_obj != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                exported_path_obj.rename(target_path)
                logger.success(f"Model moved to: {target_path}")

            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
