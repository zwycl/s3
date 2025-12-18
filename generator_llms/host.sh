#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7

while true; do
    echo "$(date): Starting vLLM server..."

    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
        --port 8000 \
        --max-model-len 8192 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --disable-frontend-multiprocessing

    EXIT_CODE=$?
    echo "$(date): Server exited with code $EXIT_CODE. Restarting in 5 seconds..."

    # Kill any zombie processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    sleep 5
done
