"""
VisionPipeline Main Entry Point.

This module orchestrates the video analysis pipeline:
1. Parse CLI arguments
2. Load configuration
3. Prepare model (download/export if needed)
4. Start multiprocessing pipeline

Pipeline Architecture:
    VideoReader → Inference → Viewer
    (Process 1)  (Process 2)  (Process 3)
"""

import multiprocessing as mp
import sys
from typing import Any

from loguru import logger

from src.core.pipeline import inference_process, video_reader_process, viewer_process
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.model_manager import ModelManager


def setup_logging(verbose: bool = False) -> None:
    """
    Configure loguru logging.

    Args:
        verbose: If True, enable DEBUG level logging
    """
    logger.remove()  # Remove default handler

    # Console handler with colors
    log_format = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    log_level = "DEBUG" if verbose else "INFO"

    logger.add(sys.stderr, format=log_format, level=log_level)


def main() -> None:
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger.info("🚀 Starting VisionPipeline...")

    # =========================================================================
    # 1. Parse CLI Arguments
    # =========================================================================
    args = parse_args()

    # =========================================================================
    # 2. Load Configuration
    # =========================================================================
    try:
        config = load_config(args.config)
        logger.success(f"Config loaded: {args.config}")
    except Exception as e:
        logger.error(f"Config error: {e}")
        return

    # =========================================================================
    # 3. Prepare Model Settings
    # =========================================================================
    model = args.model or config.get("models.default_model", "yolo")
    backend = args.backend or config.get("inference.default_backend", "pytorch")
    weights_path = None

    if backend == "triton":
        weights_path = "triton_server_model"  # Dummy path for factory
        logger.info("Triton backend selected. Skipping local weights download.")
    else:
        if args.weights:
            weights_path = args.weights
        else:
            try:
                weights_path = config.get_model_path(model, backend)
            except KeyError as e:
                logger.error(f"Config path error: {e}")
                return

        manager = ModelManager(config)
        try:
            if weights_path is None:
                logger.error("Weights path not specified")
                return
            manager.ensure_model(weights_path, model, backend)
        except Exception as e:
            logger.error(f"Model preparation failed: {e}")
            return

    # =========================================================================
    # 4. Prepare Detector Arguments
    # =========================================================================
    # Get class IDs from config (null = all classes)
    class_ids = config.get("inference.class_ids", None)

    # Log class filtering
    if class_ids:
        from src.constants import get_class_name

        class_names = [get_class_name(cid) for cid in class_ids]
        logger.info(f"🎯 Detecting classes: {class_names}")
    else:
        logger.info("🎯 Detecting all 80 COCO classes")

    detector_args = {
        "model": model,
        "backend": backend,
        "weights_path": weights_path,
        "conf_threshold": config.get("inference.conf_threshold", 0.25),
        "nms_threshold": config.get("inference.nms_threshold", 0.45),
        "input_size": tuple(config.get("inference.input_size.fixed_size", [640, 640])),
        "use_gpu": True,
        "class_ids": class_ids,
    }

    # =========================================================================
    # 5. Create Multiprocessing Queues
    # =========================================================================
    queue_frames: mp.Queue[Any] = mp.Queue(maxsize=20)
    queue_results: mp.Queue[Any] = mp.Queue(maxsize=20)

    logger.info("Spawning processes...")

    # =========================================================================
    # 6. Create and Start Processes
    # =========================================================================
    # Video Reader Process
    p_reader = mp.Process(
        target=video_reader_process, args=(args.source, queue_frames, args.max_frames)
    )

    # Inference Process
    p_inference = mp.Process(
        target=inference_process, args=(queue_frames, queue_results, detector_args)
    )

    # Get FPS from source video
    import cv2

    cap_temp = cv2.VideoCapture(args.source)
    fps_source = int(cap_temp.get(cv2.CAP_PROP_FPS)) or 30
    cap_temp.release()

    # Viewer Process
    p_viewer = mp.Process(
        target=viewer_process, args=(queue_results, args.show, args.output, fps_source)
    )

    # Start all processes
    p_reader.start()
    p_inference.start()
    p_viewer.start()

    # =========================================================================
    # 7. Wait for Completion
    # =========================================================================
    p_reader.join()
    p_inference.join()
    p_viewer.join()

    logger.success("Pipeline finished successfully!")


if __name__ == "__main__":
    mp.freeze_support()
    main()
