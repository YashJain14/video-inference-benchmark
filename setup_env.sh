#!/bin/bash
# ---------------------------------------------------------------------------
# Setup Script: Run this on the LOGIN NODE
# ---------------------------------------------------------------------------

echo "Starting environment setup..."

# 1. Conda Setup
CONDA_DIR="$HOME/miniconda3"
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo "Error: Conda not found at $CONDA_DIR"
    exit 1
fi

# 2. Create and activate a fresh conda environment
ENV_NAME="video_benchmark_env"
# echo "Creating Conda environment: $ENV_NAME"
# conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

# 3. Install Python dependencies (as specified in README)
echo "Installing pip dependencies..."
pip install protobuf

# Core dependencies (PyTorch for CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics matplotlib numpy

# Pipeline-specific dependencies
pip install nvidia-dali-cuda120
pip install torchcodec --index-url https://download.pytorch.org/whl/cu124
pip install pynvvideocodec
pip install cvcuda-cu12

# 4. Prepare Assets (Video and Model)
echo "Preparing assets..."

MODEL_FILE="yolov8n.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading YOLO model..."
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
else
    echo "Found existing model: $MODEL_FILE"
fi

VIDEO_URL="https://download.blender.org/demo/movies/ToS/ToS-4k-1920.mov"
VIDEO_FILE="ToS-4k-1920.mov"
if [ ! -f "$VIDEO_FILE" ]; then
    echo "Downloading video: $VIDEO_FILE"
    wget -q "$VIDEO_URL"
else
    echo "Found existing video: $VIDEO_FILE"
fi

echo ""
echo "Setup complete! You can now submit the job using: qsub run_benchmark.pbs"
qsub run_benchmark.pbs