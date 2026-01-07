#!/bin/bash

# Build sharded PLAID index using all available GPUs in parallel
# Each GPU builds one shard at a time

embeddings_dir=/home/ubuntu/s3/data/.temp_wiki18-mxbai-edge-colbert
shards_dir=/home/ubuntu/s3/data/wiki18-mxbai-edge-colbert
log_file=/home/ubuntu/s3/data/build_sharded_index.log

echo "=============================================="
echo "Building Sharded PLAID index from embeddings"
echo "Embeddings: $embeddings_dir"
echo "Shards dir: $shards_dir"
echo "Mode: PARALLEL (one shard per GPU)"
echo "Settings: nbits=2, kmeans_niters=4"
echo "Log file: $log_file"
echo "=============================================="

# Use --parallel to build shards across all GPUs simultaneously
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u s3/plaid_search/build_sharded_index_from_embeddings.py \
    --embeddings_dir $embeddings_dir \
    --shards_dir $shards_dir \
    --batches_per_shard 5 \
    --use_gpu \
    --parallel \
    --nbits 2 \
    --kmeans_niters 4 \
    2>&1 | tee $log_file

echo "Log saved to: $log_file"
