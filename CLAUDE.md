# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGSfM (Visual Geometry Grounded Deep Structure From Motion) is a deep learning-based 3D reconstruction system that reconstructs camera poses and 3D point clouds from images. Developed by Meta AI Research and University of Oxford's VGG group.

Key capabilities:
- Reconstructs up to 400 frames in sparse mode, 1000+ frames in sequential/video mode
- Outputs COLMAP format (cameras.bin, images.bin, points3D.bin) compatible with NeRF/Gaussian Splatting
- Supports dynamic object masking, dense depth prediction, and dense point cloud generation

## Development Setup

### Installation

```bash
# Create conda environment with Python 3.10
conda create -n vggsfm python=3.10
conda activate vggsfm

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install the package in editable mode (installs dependencies from requirements.txt)
python -m pip install -e .
```

Core dependencies are automatically installed from `requirements.txt`. Optional dependencies (poselib, visdom, gradio, pytorch3d) can be installed as needed based on features you want to use.

For optional features:
- **poselib** (better camera estimation): Already in requirements.txt
- **visdom** (3D visualization): `pip install visdom`
- **gradio** (web visualization): `pip install gradio trimesh`
- **pytorch3d** (for visdom): `conda install pytorch3d=0.7.5 -c pytorch3d`
- **Dense depth**: `pip install scikit-learn`

### Dependencies

Core dependencies in `requirements.txt`:
- `hydra-core` - configuration management
- `pycolmap==3.10.0` - COLMAP bindings for SfM
- `poselib==2.0.2` - camera pose estimation
- `lightglue` - feature matching (installed from git)
- `scipy` - scientific computing

## Running the Code

### Basic Reconstruction (demo.py)

Process unordered images (up to 400 frames):

```bash
# Basic usage
python demo.py SCENE_DIR=examples/kitchen

# With visualization in browser (useful for remote servers)
python demo.py SCENE_DIR=/path/to/images gr_visualize=True

# Common options
python demo.py SCENE_DIR=/path/to/images \
    camera_type=SIMPLE_RADIAL \
    shared_camera=True \
    query_frame_num=6 \
    max_query_pts=4096 \
    query_method=sp+sift
```

Images must be in `SCENE_DIR/images/`. Output is saved to `SCENE_DIR/sparse/` in COLMAP format.

### Sequential/Video Reconstruction (video_demo.py)

Process ordered frames (1000+ frames supported):

```bash
python video_demo.py SCENE_DIR=/path/to/video_frames
```

Images must be ordered (e.g., `0000.png`, `0001.png`, `0002.png`).

### Configuration System (Hydra)

- Main configs: `vggsfm/cfgs/demo.yaml` and `vggsfm/cfgs/video_demo.yaml`
- Override via command line: `python demo.py param=value`
- Config uses `@hydra.main(config_path="pkg://vggsfm/cfgs")` - configs are in the package

Important flags:
- `fine_tracking`: Default True, set False for faster coarse-only matching
- `shared_camera`: Set True if all frames from same camera (common for videos)
- `camera_type`: "SIMPLE_PINHOLE" (default) or "SIMPLE_RADIAL"
- `query_method`: "aliked" (default), "sp", "sift", or combinations like "sp+sift"
- `extra_pt_pixel_interval`: Set >0 to generate denser point clouds (e.g., 10)

## Architecture

### Core Model (VGGSfM)

The main model (`vggsfm/models/vggsfm.py`) has three components:

1. **TrackerPredictor** (`track_predictor.py`) - Predicts 2D point tracks across frames
   - COARSE: Uses `BasicEncoder` for feature extraction (stride 4, down_ratio 2)
   - FINE: Uses `ShallowEncoder` for refinement (optional, enabled by `fine_tracking`)

2. **CameraPredictor** (`camera_predictor.py`) - Estimates camera intrinsics and extrinsics

3. **Triangulator** (`triangulator.py`) - Triangulates 3D points from 2D tracks

### Pipeline Flow (VGGSfMRunner)

Main execution flow in `vggsfm/runners/runner.py`:

1. **Data Loading** - `DemoLoader` loads images from `SCENE_DIR/images/`
2. **Query Point Selection** - Extract keypoints using SuperPoint/SIFT/ALIKED
3. **Track Prediction** - Predict 2D tracks across all frames
4. **Preliminary Cameras** - Estimate initial camera poses (uses poselib if available)
5. **Camera Refinement** - Refine camera parameters
6. **Triangulation** - Generate 3D points from tracks
7. **Bundle Adjustment** - Joint optimization of cameras and points (via pycolmap)
8. **Output** - Save to COLMAP format in `SCENE_DIR/sparse/`

For video mode (`video_runner.py`), uses sliding window approach with periodic joint bundle adjustment.

### Memory Management

Two critical parameters for GPU memory (hardcoded in code):
- `max_points_num` in `predict_tracks` (default: 163840 for 32GB GPU)
- `max_tri_points_num` in `triangulate_tracks` (default: 819200 for 32GB GPU)

Located in:
- `vggsfm/runners/runner.py` - predict_tracks function
- `vggsfm/utils/triangulation.py` - triangulate_tracks function

Reduce these if encountering OOM errors on smaller GPUs.

### Key Utilities

- `vggsfm/utils/triangulation.py` - Core triangulation logic, chunked processing
- `vggsfm/utils/tensor_to_pycolmap.py` - Convert tensors to COLMAP format
- `vggsfm/utils/visualizer.py` - Visdom visualization
- `vggsfm/utils/gradio.py` - Gradio web interface
- `vggsfm/two_view_geo/estimate_preliminary.py` - Initial camera estimation

### Data Format

Input:
- Images in `SCENE_DIR/images/`
- Optional masks in `SCENE_DIR/masks/` (binary: 1=filter out, 0=keep)

Output (COLMAP format in `SCENE_DIR/sparse/`):
- `cameras.bin` - Camera intrinsics
- `images.bin` - Camera extrinsics (poses)
- `points3D.bin` - 3D point cloud

## Optional Features

### Dynamic Object Masking

Place binary masks in `SCENE_DIR/masks/` with same filenames as images. Value 1 = filter out (dynamic), 0 = keep (static).

### Dense Depth Maps

Requires Depth-Anything-V2:

```bash
pip install scikit-learn
git clone git@github.com:DepthAnything/Depth-Anything-V2.git dependency/depth_any_v2
python -m pip install -e .
python demo.py dense_depth=True
```

### Denser Point Clouds

```bash
python demo.py extra_pt_pixel_interval=10 concat_extra_points=True
```

### Gaussian Splatting Training

After reconstruction, use gsplat:

```bash
cd gsplat
python examples/simple_trainer.py default --data_factor 1 \
    --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```

## Model Checkpoints

Checkpoints auto-download from HuggingFace (facebook/VGGSfM) on first run. Manual download available from:
- https://huggingface.co/facebook/VGGSfM/blob/main/vggsfm_v2_0_0.bin

To use local checkpoint: `auto_download_ckpt=False resume_ckpt=/path/to/checkpoint.bin`

## Package Structure

```
vggsfm/
├── cfgs/               # Hydra configuration files
│   ├── demo.yaml
│   └── video_demo.yaml
├── datasets/           # Data loading
│   └── demo_loader.py
├── models/             # Neural network models
│   ├── vggsfm.py       # Main model container
│   ├── track_predictor.py
│   ├── camera_predictor.py
│   ├── triangulator.py
│   └── track_modules/  # Track prediction submodules
├── runners/            # Main execution logic
│   ├── runner.py       # Standard reconstruction
│   └── video_runner.py # Sequential/video reconstruction
├── utils/              # Utilities
│   ├── triangulation.py
│   ├── tensor_to_pycolmap.py
│   ├── visualizer.py
│   └── gradio.py
└── two_view_geo/       # Two-view geometry estimation

dependency/
├── LightGlue/          # Feature matching (git submodule)
└── depth_any_v2/       # Dense depth (optional)
```

## Entry Points

The package defines two console scripts in setup.py:
- `vggsfm-image` → `vggsfm.demo:demo_fn`
- `vggsfm-video` → `vggsfm.video_demo:demo_fn`

Direct execution uses the module files in the package.
