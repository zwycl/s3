"""
Build sharded PLAID indexes - multiple batches per shard.
Each shard is built fresh on GPU (no OOM from updates).
Search queries all shards and merges results.
"""
import os
import argparse
import time
import gc
import torch
from tqdm import tqdm
from pylate import indexes
import numpy as np
from typing import List, Optional, Union


def build_sharded_index_from_embedding(
    documents_ids: List[str],
    documents_embeddings: List[np.ndarray],
    shards_dir: str,
    docs_per_shard: int = 100000,
    device: Union[str, List[str]] = "cuda:0",
    nbits: int = 4,
    kmeans_niters: int = 4,
    use_fast: bool = True,
    override: bool = False,
    show_progress: bool = True,
) -> str:
    """
    Build a sharded PLAID index from pre-computed embeddings.

    This creates multiple shard indexes that can be searched using ShardedRetriever.

    Args:
        documents_ids: List of document IDs
        documents_embeddings: List of document embeddings (each is a 2D array of token embeddings)
        shards_dir: Directory to store the sharded indexes (will contain shard_0000, shard_0001, etc.)
        docs_per_shard: Number of documents per shard
        device: Device(s) for index building (e.g., "cuda:0" or ["cuda:0", "cuda:1"])
        nbits: Bits for quantization (2=fastest, 4=balanced, 8=highest quality)
        kmeans_niters: K-means iterations (2=fast, 4=balanced, 10+=high quality)
        use_fast: Use FastPLAID backend
        override: Whether to override existing shards
        show_progress: Whether to show progress bars

    Returns:
        Path to the shards directory
    """
    total_start = time.time()

    num_docs = len(documents_ids)
    num_shards = (num_docs + docs_per_shard - 1) // docs_per_shard

    print(f"Building sharded index with {num_docs:,} documents")
    print(f"Will create {num_shards} shards ({docs_per_shard:,} docs per shard)")
    print(f"Shards directory: {shards_dir}")

    # Create shards directory
    os.makedirs(shards_dir, exist_ok=True)

    # Build each shard
    for shard_idx in range(num_shards):
        shard_name = f"shard_{shard_idx:04d}"
        shard_path = os.path.join(shards_dir, shard_name)

        # Skip if already exists and not overriding
        if os.path.exists(shard_path) and not override:
            print(f"Shard {shard_idx} already exists, skipping")
            continue

        # Get documents for this shard
        start_idx = shard_idx * docs_per_shard
        end_idx = min(start_idx + docs_per_shard, num_docs)

        shard_doc_ids = documents_ids[start_idx:end_idx]
        shard_embeddings = documents_embeddings[start_idx:end_idx]

        print(f"\n{'='*50}")
        print(f"Building shard {shard_idx}: documents {start_idx:,}-{end_idx-1:,} ({len(shard_doc_ids):,} docs)")
        print(f"{'='*50}")
        start = time.time()

        # Create fresh index for this shard
        index = indexes.PLAID(
            index_folder=shards_dir,
            index_name=shard_name,
            override=True,
            show_progress=show_progress,
            device=device,
            use_fast=use_fast,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
        )

        index.add_documents(
            documents_ids=shard_doc_ids,
            documents_embeddings=shard_embeddings,
        )

        # Clear memory
        del index
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Shard {shard_idx} built in {time.time() - start:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Sharded index built successfully!")
    print(f"Total documents: {num_docs:,}")
    print(f"Number of shards: {num_shards}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Shards location: {shards_dir}")
    print(f"{'='*50}")

    return shards_dir


def build_sharded_index_from_embedding_files(
    embeddings_dir: str,
    shards_dir: str,
    batches_per_shard: int = 3,
    shard_id: int = -1,
    device: Union[str, List[str]] = "cuda:0",
    nbits: int = 4,
    kmeans_niters: int = 4,
    use_fast: bool = True,
    show_progress: bool = True,
) -> str:
    """
    Build a sharded PLAID index from saved embedding files (.npz).

    This loads embeddings from .npz files and creates multiple shard indexes
    that can be searched using ShardedRetriever.

    Args:
        embeddings_dir: Directory containing saved embedding batches (.npz files)
        shards_dir: Directory to store the sharded indexes
        batches_per_shard: Number of embedding batches per shard
        shard_id: Specific shard to build (-1 for all shards)
        device: Device(s) for index building
        nbits: Bits for quantization
        kmeans_niters: K-means iterations
        use_fast: Use FastPLAID backend
        show_progress: Whether to show progress bars

    Returns:
        Path to the shards directory
    """
    total_start = time.time()

    # Get batch files
    batch_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npz')])
    print(f"Found {len(batch_files)} embedding batches")

    # Group batches into shards
    num_shards = (len(batch_files) + batches_per_shard - 1) // batches_per_shard
    print(f"Will create {num_shards} shards ({batches_per_shard} batches each)")

    # Create shards directory
    os.makedirs(shards_dir, exist_ok=True)

    # Determine which shards to build
    if shard_id >= 0:
        shard_ids = [shard_id]
    else:
        shard_ids = range(num_shards)

    # Build each shard
    for shard_idx in shard_ids:
        shard_name = f"shard_{shard_idx:04d}"
        shard_path = os.path.join(shards_dir, shard_name)

        # Skip if already exists
        if os.path.exists(shard_path):
            print(f"Shard {shard_idx} already exists, skipping")
            continue

        # Get batch files for this shard
        start_batch = shard_idx * batches_per_shard
        end_batch = min(start_batch + batches_per_shard, len(batch_files))
        shard_batch_files = batch_files[start_batch:end_batch]

        print(f"\n{'='*50}")
        print(f"Building shard {shard_idx}: batches {start_batch}-{end_batch-1} ({len(shard_batch_files)} batches)")
        print(f"{'='*50}")
        start = time.time()

        # Load and combine all embeddings for this shard
        all_doc_ids = []
        all_embeddings = []

        iterator = tqdm(shard_batch_files, desc=f"Loading batches for shard {shard_idx}") if show_progress else shard_batch_files
        for batch_file in iterator:
            batch_path = os.path.join(embeddings_dir, batch_file)
            data = np.load(batch_path, allow_pickle=True)
            all_doc_ids.extend(data['ids'].tolist())
            all_embeddings.extend(data['embeddings'].tolist())
            del data
            gc.collect()

        print(f"Loaded {len(all_doc_ids):,} documents for shard {shard_idx}")

        # Create fresh index for this shard
        print(f"Building PLAID index on {device}...")
        index = indexes.PLAID(
            index_folder=shards_dir,
            index_name=shard_name,
            override=True,
            show_progress=show_progress,
            device=device,
            use_fast=use_fast,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
        )

        index.add_documents(
            documents_ids=all_doc_ids,
            documents_embeddings=all_embeddings,
        )

        # Clear memory
        del all_doc_ids, all_embeddings, index
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Shard {shard_idx} built in {time.time() - start:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Sharded index built successfully!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Shards location: {shards_dir}")
    print(f"Number of shards: {num_shards}")
    print(f"{'='*50}")

    return shards_dir


def main():
    parser = argparse.ArgumentParser(description="Build sharded PLAID indexes")

    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing saved embedding batches')
    parser.add_argument('--index_name', type=str, default='wiki18-colbert',
                        help='Base name for the output indexes')
    parser.add_argument('--index_folder', type=str, default='/home/ubuntu/s3/data',
                        help='Folder to store the indexes')
    parser.add_argument('--batches_per_shard', type=int, default=3,
                        help='Number of embedding batches per shard')
    parser.add_argument('--shard_id', type=int, default=-1,
                        help='Specific shard to build (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for index building')
    parser.add_argument('--pool_factor', type=int, default=1,
                        help='Pool neighboring token embeddings (2=halves index size, 1=no pooling)')

    args = parser.parse_args()

    total_start = time.time()

    # Get batch files
    batch_files = sorted([f for f in os.listdir(args.embeddings_dir) if f.endswith('.npz')])
    print(f"Found {len(batch_files)} embedding batches")

    # Group batches into shards
    num_shards = (len(batch_files) + args.batches_per_shard - 1) // args.batches_per_shard
    print(f"Will create {num_shards} shards ({args.batches_per_shard} batches each)")

    # Create shards directory
    shards_dir = os.path.join(args.index_folder, f"{args.index_name}_shards")
    os.makedirs(shards_dir, exist_ok=True)

    # Determine which shards to build
    if args.shard_id >= 0:
        shard_ids = [args.shard_id]
    else:
        shard_ids = range(num_shards)

    # Build each shard
    for shard_idx in shard_ids:
        shard_name = f"shard_{shard_idx:04d}"
        shard_path = os.path.join(shards_dir, shard_name)

        # Skip if already exists
        if os.path.exists(shard_path):
            print(f"Shard {shard_idx} already exists, skipping")
            continue

        # Get batch files for this shard
        start_batch = shard_idx * args.batches_per_shard
        end_batch = min(start_batch + args.batches_per_shard, len(batch_files))
        shard_batch_files = batch_files[start_batch:end_batch]

        print(f"\n{'='*50}")
        print(f"Building shard {shard_idx}: batches {start_batch}-{end_batch-1} ({len(shard_batch_files)} batches)")
        print(f"{'='*50}")
        start = time.time()

        # Load and combine all embeddings for this shard
        all_doc_ids = []
        all_embeddings = []

        for batch_file in tqdm(shard_batch_files, desc=f"Loading batches for shard {shard_idx}"):
            batch_path = os.path.join(args.embeddings_dir, batch_file)
            data = np.load(batch_path, allow_pickle=True)
            all_doc_ids.extend(data['ids'].tolist())
            all_embeddings.extend(data['embeddings'].tolist())
            del data
            gc.collect()

        print(f"Loaded {len(all_doc_ids):,} documents for shard {shard_idx}")

        # Create fresh index for this shard
        print(f"Building PLAID index on {args.device}...")
        index = indexes.PLAID(
            index_folder=shards_dir,
            index_name=shard_name,
            override=True,
            show_progress=True,
            device=args.device,
        )

        index.add_documents(
            documents_ids=all_doc_ids,
            documents_embeddings=all_embeddings,
        )

        # Clear memory
        del all_doc_ids, all_embeddings, index
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Shard {shard_idx} built in {time.time() - start:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Sharded index built successfully!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Shards location: {shards_dir}")
    print(f"Number of shards: {len(batch_files)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
