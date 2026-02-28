# VisionPipeline

Video Analytics Pipeline with Multi-Backend Inference Support

## Features

- **Multi-Backend Support**: PyTorch, ONNX Runtime, TensorRT, NVIDIA Triton
- **Multiple Architectures**: YOLO (v8, v11) and RT-DETR
- **80 COCO Classes**: Full object detection with configurable class filtering
- **Multi-Process Pipeline**: VideoReader → Inference → Viewer (bypasses Python GIL)
- **Docker Support**: Containerized deployment with Docker Compose

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/VisionPipeline.git
cd VisionPipeline

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/Mac

# Full installation (recommended)
pip install -e ".[all]"

# Or selective installation:
pip install -e ".[pytorch]"      # PyTorch + TensorRT backend
pip install -e ".[onnx]"         # ONNX Runtime only
pip install -e ".[triton]"       # Triton server client
pip install -e ".[dev,pytorch]"  # Development with PyTorch
```

### GPU Support (CUDA 13.0)

```bash
# PyTorch with CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[pytorch]"

# ONNX Runtime with CUDA
pip install onnxruntime-gpu
```

## Quick Start

```bash
# Basic inference
python main.py --source video.mp4 --backend pytorch --model yolo

# With TensorRT (fastest)
python main.py --source video.mp4 --backend tensorrt --model yolo

# With ONNX
python main.py --source video.mp4 --backend onnx --model yolo
```

## CLI Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | **required** | Path to input video file |
| `--output` | str | `None` | Path to output video |
| `--config` | str | `config.yaml` | Path to config file |
| `--model` | str | `yolo` | Model: `yolo`, `rtdetr`, `rtdetr_v2` |
| `--backend` | str | `pytorch` | Backend: `pytorch`, `onnx`, `tensorrt`, `triton` |
| `--weights` | str | from config | Path to model weights |
| `--conf` | float | `0.25` | Confidence threshold |
| `--nms` | float | `0.45` | NMS IoU threshold |
| `--max-frames` | int | `None` | Maximum frames to process |
| `--show` | flag | `False` | Show preview window |
| `--save-metrics` | str | `None` | Save metrics to JSON file |
| `--no-metrics` | flag | `False` | Disable metrics collection |

### Examples

```bash
# Basic usage
python main.py --source video.mp4

# With TensorRT and preview
python main.py --source video.mp4 --backend tensorrt --show

# Save output video
python main.py --source video.mp4 --output result.mp4

# Process first 100 frames with custom threshold
python main.py --source video.mp4 --max-frames 100 --conf 0.5

# RT-DETR with ONNX
python main.py --source video.mp4 --model rtdetr --backend onnx

# Custom weights path
python main.py --source video.mp4 --weights ./custom_model.pt
```

## Backends

| Backend | Install | Weights | Best For |
|---------|---------|---------|----------|
| PyTorch | `.[pytorch]` | `.pt` | Development, flexibility |
| TensorRT | `.[pytorch]` | `.engine` | Maximum GPU performance |
| ONNX | `.[onnx]` | `.onnx` | Cross-platform deployment |
| Triton | `.[triton]` | server | Production deployment |

> **Note:** ONNX auto-export requires `ultralytics` if `.onnx` file doesn't exist.

## Docker

```bash
# Build and run with Triton server
docker-compose up --build

# Or run inference only
docker build -t visionpipeline .
```

## Configuration

Edit `config.yaml`:

```yaml
inference:
  conf_threshold: 0.25
  nms_threshold: 0.45

  # Class filtering
  class_ids: null    # null = all 80 COCO classes
                     # [0] = persons only
                     # [0, 2, 3, 5, 7] = persons + vehicles
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run checks manually
pre-commit run --all-files

# Or individually:
ruff check src/          # Linting
ruff format src/         # Formatting
mypy src/                # Type checking
```

## Project Structure

```
src/
├── constants/      # COCO classes and colors
├── core/           # Multi-process pipeline
├── detectors/      # Detector implementations
│   ├── base_detector.py
│   ├── ultralytics_detector.py
│   ├── onnx_detector.py
│   └── triton_detector.py
├── export/         # Model export utilities
├── types/          # Type definitions (TypedDict, Protocol)
└── utils/          # Config, CLI, helpers
```

## Requirements

- Python 3.12+
- NVIDIA GPU + CUDA 12.x/13.x (for GPU backends)
- Docker (optional, for containerized deployment)

## License

MIT
