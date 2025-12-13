#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check and install CUDA Toolkit if not present (required for flash-attn compilation)
if ! command -v nvcc &> /dev/null; then
    echo "CUDA Toolkit not found. Installing CUDA Toolkit 12.6..."

    # Download and install NVIDIA CUDA keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt update

    # Install CUDA toolkit
    sudo apt install -y cuda-toolkit-12-6

    # Set environment variables for current session
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    # Add to bashrc for future sessions
    if ! grep -q "CUDA_HOME=/usr/local/cuda-12.6" ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'

# CUDA Toolkit 12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    fi

    echo "CUDA Toolkit 12.6 installed successfully."
else
    echo "CUDA Toolkit already installed: $(nvcc --version | grep release)"
    # Ensure CUDA_HOME is set
    if [ -z "$CUDA_HOME" ]; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    fi
fi

# Install uv package manager
pip install uv

# Install torch first with CUDA 12.1 support (required for flash-attn build)
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn build dependencies
uv pip install psutil numpy ninja packaging

# Install core training requirements
uv pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    pandas \
    pybind11 \
    ray \
    'tensordict<0.6' \
    'transformers<4.48' \
    'vllm<=0.6.3' \
    wandb \
    IPython \
    matplotlib \
    huggingface_hub

# Install retriever dependencies
uv pip install \
    faiss-gpu \
    fastapi \
    uvicorn \
    pyserini

# Install flash-attn (needs torch available, so no build isolation)
uv pip install flash-attn --no-build-isolation

# Install verl package in editable mode
uv pip install -e "$SCRIPT_DIR"

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Download data: python scripts/download.py --save_path ./data"
echo "2. Concatenate index: cat data/part_* > data/e5_Flat.index"
echo "3. Extract corpus: gzip -d data/wiki-18.jsonl.gz"
