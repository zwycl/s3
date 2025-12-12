#!/bin/bash
set -e

# Install uv package manager
pip install uv

# Install torch first (required for flash-attn build)
uv pip install torch

# Install flash-attn build dependencies
uv pip install psutil numpy ninja packaging

# Install remaining requirements (flash-attn excluded)
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
    matplotlib

# Install flash-attn (needs torch available, so no build isolation)
uv pip install flash-attn --no-build-isolation
