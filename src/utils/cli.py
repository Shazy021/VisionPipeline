"""CLI argument parser."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Person detection in video with configurable backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")

    # Configuration
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    # Model
    parser.add_argument("--model", type=str, default=None, help="Model architecture")
    parser.add_argument("--backend", type=str, default=None, help="Inference backend")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")

    # Inference parameters
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=None, help="NMS threshold")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override input size, e.g. --input-size 640 640",
    )

    # Processing
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser.parse_args()
