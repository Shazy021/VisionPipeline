"""
VisionPipeline Main Entry Point.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

from loguru import logger

from src.utils.cli import parse_args
from src.utils.container import Container


def setup_logging(verbose: bool = False) -> None:
    """Настроить логирование."""
    logger.remove()
    log_format = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=log_format, level=log_level)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(getattr(args, "verbose", False))

    container = Container()
    container.register_config(args.config)
    container.register_cli_args(args)

    try:
        orchestrator = container.get_orchestrator()
        orchestrator.run()

    except KeyboardInterrupt:
        logger.warning("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        logger.success("👋 Done.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
