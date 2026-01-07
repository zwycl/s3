#!/usr/bin/env python
"""Worker script for building a single PLAID shard."""
import os
import sys
import argparse
import time
import gc
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--shard_idx', type=int, required=True)
parser.add_argument('--batch_files', type=str, required=True)
parser.add_argument('--embeddings_dir', type=str, required=True)
parser.add_argument('--shards_dir', type=str, required=True)
parser.add_argument('--nbits', type=int, default=2)
parser.add_argument('--kmeans_niters', type=int, default=4)
args = parser.parse_args()

import torch
from pylate import indexes

def log(msg):
    print(msg, flush=True)

gpu_id = args.gpu_id
shard_idx = args.shard_idx
batch_files = args.batch_files.split(',')

log(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
log(f"[GPU {gpu_id}] torch.cuda.device_count()={torch.cuda.device_count()}")

try:
    device = "cuda:0"
    shard_name = f"shard_{shard_idx:04d}"
    start_time = time.time()

    log(f"[GPU {gpu_id}] Starting shard {shard_idx} ({len(batch_files)} batches)")

    index = indexes.PLAID(
        index_folder=args.shards_dir,
        index_name=shard_name,
        override=True,
        show_progress=True,
        device=device,
        use_fast=True,
        nbits=args.nbits,          
        kmeans_niters=args.kmeans_niters,    
        use_triton=False,
    )

    num_docs = 0
    for i, batch_file in enumerate(batch_files):
        batch_path = os.path.join(args.embeddings_dir, batch_file)
        data = np.load(batch_path, allow_pickle=True)
        doc_ids = data['ids'].tolist()
        embeddings = data['embeddings'].tolist()
        del data

        log(f"[GPU {gpu_id}] Shard {shard_idx}: Batch {i+1}/{len(batch_files)} - adding {len(doc_ids):,} docs...")
        index.add_documents(documents_ids=doc_ids, documents_embeddings=embeddings)
        num_docs += len(doc_ids)

        del doc_ids, embeddings
        gc.collect()
        torch.cuda.empty_cache()

    del index
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    log(f"[GPU {gpu_id}] Shard {shard_idx} COMPLETED in {elapsed:.1f}s ({num_docs:,} docs)")
    sys.exit(0)

except Exception as e:
    import traceback
    log(f"[GPU {gpu_id}] Shard {shard_idx} FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
