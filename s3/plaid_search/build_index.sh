#!/bin/bash

corpus_file=/home/ubuntu/s3/data/wiki_dump.jsonl
checkpoint=mixedbread-ai/mxbai-edge-colbert-v0-17m
log_file=/home/ubuntu/s3/data/build_index.log
embeddings_dir=/home/ubuntu/s3/data/.temp_wiki18-mxbai-edge-colbert

echo "=============================================="
echo "Starting PLAID index build"
echo "Corpus: $corpus_file"
echo "Embeddings dir: $embeddings_dir"
echo "Checkpoint: $checkpoint"
echo "Log file: $log_file"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u s3/plaid_search/index_builder.py \
    --corpus_path $corpus_file \
    --embeddings_dir $embeddings_dir \
    --checkpoint $checkpoint \
    --max_document_length 256 \
    --batch_size 512 \
    --mega_batch_size 1000000 \
    --pool_factor 1 \
    2>&1 | tee $log_file

echo "Log saved to: $log_file"
