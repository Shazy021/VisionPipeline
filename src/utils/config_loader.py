from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

if TYPE_CHECKING:
    from argparse import Namespace


class Config:
    """
    Configuration manager with support for YAML files and CLI overrides.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path) as f:
            self.data = yaml.safe_load(f)

        self._validate()

    def _validate(self) -> None:
        """Validate config structure."""
        required_sections = ["models", "inference", "video", "export", "metrics"]
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required config section: {section}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = key_path.split(".")
        value = self.data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_model_path(self, model: str, backend: str) -> str:
        """Get model weights path."""
        try:
            path: str = self.data["models"][model][backend]
            return path
        except KeyError:
            raise KeyError(f"Model path not found: models.{model}.{backend}") from None

    def merge_cli_args(self, args: Namespace) -> None:
        """Merge CLI arguments into config (CLI takes priority)."""
        cli_mappings = {
            "conf": "inference.conf_threshold",
            "nms": "inference.nms_threshold",
            "max_frames": "video.max_frames",
            "show": "video.show_preview",
            "no_display_info": "video.display_info",
        }

        for cli_key, config_path in cli_mappings.items():
            if hasattr(args, cli_key):
                cli_value = getattr(args, cli_key)
                if cli_value is None:
                    continue
                if cli_key == "no_display_info":
                    cli_value = not cli_value
                self._set_nested(config_path, cli_value)

    def _set_nested(self, key_path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key_path.split(".")
        d = self.data
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def print_summary(self) -> None:
        """Print configuration summary using loguru."""
        logger.info("=" * 60)
        logger.info("Configuration Summary")
        logger.info("=" * 60)
        logger.info(f"Config file: {self.config_path}")
        logger.info("Inference:")
        logger.info(f"  Confidence threshold: {self.get('inference.conf_threshold')}")
        logger.info(f"  NMS threshold: {self.get('inference.nms_threshold')}")
        logger.info(f"  GPU enabled: {self.get('inference.device.use_gpu')}")
        logger.info("Video:")
        logger.info(f"  Show preview: {self.get('video.show_preview')}")
        logger.info(f"  Max frames: {self.get('video.max_frames') or 'all'}")
        logger.info("=" * 60 + "\n")


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    return Config(config_path)
