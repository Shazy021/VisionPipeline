"""
VisionPipeline Main Entry Point.

This module orchestrates the video analysis pipeline:
1. Parse CLI arguments
2. Load configuration
3. Prepare model (download/export if needed)
4. Resolve input size
5. Start multiprocessing pipeline

Pipeline Architecture:
    VideoReader → Inference → Viewer
    (Process 1)  (Process 2)  (Process 3)
"""

from __future__ import annotations

import multiprocessing as mp
import sys

from loguru import logger

from src.core.pipeline import inference_process, video_reader_process, viewer_process
from src.types import VideoInfo
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.input_size import InputSizeResolver
from src.utils.model_manager import ModelManager
from src.utils.utils import get_video_optimal_size


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
    weights_path: str | None = None

    if backend == "triton":
        weights_path = "triton_server_model"
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
            manager.ensure_model(weights_path, backend)
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

    # =========================================================================
    # 5. Get Video Metadata
    # =========================================================================
    try:
        video_info_dict = get_video_optimal_size(args.source, None)
        video_info = VideoInfo.from_dict(video_info_dict)
    except ValueError as e:
        logger.error(str(e))
        return

    fps_source = video_info.fps or 30

    # =========================================================================
    # 6. Resolve Input Size
    # =========================================================================
    resolver = InputSizeResolver(config, video_info)

    result = resolver.resolve(
        backend=backend,
        cli_override=args.input_size,
        model=model,
        weights_path=weights_path,
    )

    input_size = result.size
    backend_url = result.backend_url

    # =========================================================================
    # 7. Build Detector Arguments
    # =========================================================================
    detector_args = {
        "model": model,
        "backend": backend,
        "weights_path": weights_path,
        "conf_threshold": config.get("inference.conf_threshold", 0.25),
        "nms_threshold": config.get("inference.nms_threshold", 0.45),
        "input_size": input_size,
        "use_gpu": True,
        "class_ids": class_ids,
    }

    if backend_url:
        detector_args["triton_url"] = backend_url

    # Get batch size from config (default: 1)
    batch_size = config.get("inference.batch_size", 1)

    # =========================================================================
    # 8. Create Multiprocessing Queues
    # =========================================================================
    queue_frames: mp.Queue[object] = mp.Queue(maxsize=20)
    queue_results: mp.Queue[object] = mp.Queue(maxsize=20)

    logger.info("Spawning processes...")

    # =========================================================================
    # 9. Create and Start Processes
    # =========================================================================
    # Video Reader Process
    p_reader = mp.Process(
        target=video_reader_process, args=(args.source, queue_frames, args.max_frames)
    )

    # Inference Process
    p_inference = mp.Process(
        target=inference_process, args=(queue_frames, queue_results, detector_args, batch_size)
    )

    # Viewer Process
    p_viewer = mp.Process(
        target=viewer_process, args=(queue_results, args.show, args.output, fps_source)
    )

    p_reader.daemon = True
    p_inference.daemon = True
    p_viewer.daemon = True

    # Start all processes
    p_reader.start()
    p_inference.start()
    p_viewer.start()

    # =========================================================================
    # 10. Wait for Completion with Graceful Shutdown
    # =========================================================================

    try:
        # Wait for reader to finish (or indefinitely for streams)
        while p_reader.is_alive():
            p_reader.join(timeout=1.0)

        # Wait for other processes
        p_inference.join(timeout=2.0)
        p_viewer.join(timeout=2.0)

    except KeyboardInterrupt:
        logger.warning("\n[Main] Stopping pipeline...")
        p_reader.terminate()
        p_inference.terminate()
        p_viewer.terminate()
    finally:
        logger.success("Pipeline stopped.")
        logger.info("👋 All processes stopped.")
        sys.exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
