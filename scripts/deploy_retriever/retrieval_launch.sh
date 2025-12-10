export CUDA_VISIBLE_DEVICES=0
file_path=/workspace/s3_data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python s3/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 12 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --port 3000 \
                                            --faiss_gpu
