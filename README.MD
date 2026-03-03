# VisionPipeline

Video Analytics Pipeline with Multi-Backend Inference Support

## Features

- **Multi-Backend Support**: PyTorch, ONNX Runtime, TensorRT, NVIDIA Triton
- **YOLO Architecture**: YOLOv8, YOLOv11 and newer
- **80 COCO Classes**: Full object detection with configurable class filtering
- **Multi-Process Pipeline**: VideoReader → Inference → Viewer (bypasses Python GIL)
- **Docker Support**: Containerized deployment with Docker Compose
- **Smart Input Size**: Auto-calculation from video or query from server metadata
- **Type-Safe**: Strict mypy configuration, TypedDict, Protocol

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

### GPU Support

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
| `--backend` | str | `pytorch` | Backend: `pytorch`, `onnx`, `tensorrt`, `triton` |
| `--weights` | str | from config | Path to model weights |
| `--conf` | float | `0.25` | Confidence threshold |
| `--nms` | float | `0.45` | NMS IoU threshold |
| `--max-frames` | int | `None` | Maximum frames to process |
| `--show` | flag | `False` | Show preview window |

## Backends

| Backend | Install | Weights | Best For |
|---------|---------|---------|----------|
| PyTorch | `.[pytorch]` | `.pt` | Development, flexibility |
| TensorRT | `.[tensorrt]` | `.engine` | Maximum GPU performance |
| ONNX | `.[onnx]` | `.onnx` | Cross-platform deployment |
| Triton | `.[triton]` | server | Production, remote inference |

## Input Size Resolution

Input size is resolved automatically with priority:

```
CLI --input-size > Backend query > Config (auto/fixed)
```

| Backend | Source |
|---------|--------|
| PyTorch | Config (auto from video / fixed) |
| TensorRT | Config (auto from video / fixed) |
| ONNX | Config or model metadata inspection |
| Triton | Server model metadata (fixed/dynamic) |

Config:
```yaml
inference:
  input_size:
    mode: auto          # auto | fixed
    fixed_size: [640, 640]
```


## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run checks
ruff check src/
mypy src/
```

## Requirements

- Python 3.12+
- NVIDIA GPU + CUDA 12.x/13.x (for GPU backends)
- Docker (optional, for containerized deployment)

## License

MIT
