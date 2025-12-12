#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
