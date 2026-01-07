#!/bin/bash

embeddings_dir=/home/ubuntu/s3/data/.temp_wiki18-colbert
index_name=wiki18-colbert
index_folder=/home/ubuntu/s3/data
log_file=/home/ubuntu/s3/data/build_index_from_embeddings.log

echo "=============================================="
echo "Building FastPLAID index from saved embeddings"
echo "Embeddings: $embeddings_dir"
echo "Index: $index_folder/$index_name"
echo "Settings: balanced (nbits=4, kmeans_niters=4)"
echo "Log file: $log_file"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u s3/plaid_search/build_index_from_embeddings.py \
    --embeddings_dir $embeddings_dir \
    --index_name $index_name \
    --index_folder $index_folder \
    --use_gpu \
    --nbits 4 \
    --kmeans_niters 4 \
    2>&1 | tee $log_file

echo "Log saved to: $log_file"
