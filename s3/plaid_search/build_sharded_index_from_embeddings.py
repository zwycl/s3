"""
Build sharded PLAID index from saved embeddings.
Creates multiple shard indexes that can be loaded by ShardedRetriever.
Supports parallel building across multiple GPUs.
"""
import os
import sys
import argparse
import time
import gc
import multiprocessing as mp
from tqdm import tqdm
from pylate import indexes
import numpy as np
import torch
from typing import List, Union


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
            use_triton=False,
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


def build_shard_worker(
    gpu_id: int,
    shard_idx: int,
    batch_files: List[str],
    embeddings_dir: str,
    shards_dir: str,
    nbits: int,
    kmeans_niters: int,
    result_queue: mp.Queue,
):
    """
    Worker function to build a single shard on a specific GPU.
    Runs in a separate process.
    """
    import sys

    def log(msg):
        """Print with flush to ensure output is visible in parallel mode."""
        print(msg, flush=True)
        sys.stdout.flush()

    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        shard_name = f"shard_{shard_idx:04d}"
        shard_path = os.path.join(shards_dir, shard_name)

        start_time = time.time()
        log(f"[GPU {gpu_id}] Starting shard {shard_idx} ({len(batch_files)} batches)")

        # Load embeddings for this shard
        all_doc_ids = []
        all_embeddings = []

        for i, batch_file in enumerate(batch_files):
            batch_path = os.path.join(embeddings_dir, batch_file)
            data = np.load(batch_path, allow_pickle=True)
            all_doc_ids.extend(data['ids'].tolist())
            all_embeddings.extend(data['embeddings'].tolist())
            del data
            gc.collect()
            log(f"[GPU {gpu_id}] Shard {shard_idx}: Loaded batch {i+1}/{len(batch_files)}")

        num_docs = len(all_doc_ids)
        log(f"[GPU {gpu_id}] Shard {shard_idx}: Loaded {num_docs:,} documents total, building index...")

        # Build index
        index = indexes.PLAID(
            index_folder=shards_dir,
            index_name=shard_name,
            override=True,
            show_progress=True,  # Enable progress bar
            device=device,
            use_fast=True,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            use_triton=False,
        )

        log(f"[GPU {gpu_id}] Shard {shard_idx}: Adding documents to index...")

        index.add_documents(
            documents_ids=all_doc_ids,
            documents_embeddings=all_embeddings,
        )

        # Cleanup
        del all_doc_ids, all_embeddings, index
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        log(f"[GPU {gpu_id}] Shard {shard_idx} COMPLETED in {elapsed:.1f}s ({num_docs:,} docs)")

        result_queue.put({
            "status": "success",
            "gpu_id": gpu_id,
            "shard_idx": shard_idx,
            "num_docs": num_docs,
            "elapsed": elapsed,
        })

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        log(f"[GPU {gpu_id}] Shard {shard_idx} FAILED: {e}")
        log(error_msg)
        result_queue.put({
            "status": "error",
            "gpu_id": gpu_id,
            "shard_idx": shard_idx,
            "error": str(e),
            "traceback": error_msg,
        })


def build_shards_parallel(
    embeddings_dir: str,
    shards_dir: str,
    batch_files: List[str],
    batches_per_shard: int,
    num_gpus: int,
    nbits: int,
    kmeans_niters: int,
    start_shard: int = 0,
):
    """
    Build multiple shards in parallel using subprocess (one GPU per process).
    """
    import subprocess

    num_shards = (len(batch_files) + batches_per_shard - 1) // batches_per_shard

    # Get list of shards to build
    shards_to_build = []
    for shard_idx in range(start_shard, num_shards):
        shard_path = os.path.join(shards_dir, f"shard_{shard_idx:04d}")
        if os.path.exists(shard_path):
            print(f"Shard {shard_idx} already exists, skipping")
            continue

        start_batch = shard_idx * batches_per_shard
        end_batch = min(start_batch + batches_per_shard, len(batch_files))
        shard_batch_files = batch_files[start_batch:end_batch]
        shards_to_build.append((shard_idx, shard_batch_files))

    if not shards_to_build:
        print("All shards already exist!")
        return 0

    print(f"\n{'='*60}")
    print(f"PARALLEL BUILD: {len(shards_to_build)} shards across {num_gpus} GPUs")
    print(f"{'='*60}\n")

    os.makedirs(shards_dir, exist_ok=True)

    worker_script = os.path.join(os.path.dirname(__file__), "shard_worker.py")
    total_start = time.time()
    active_procs = {}  # gpu_id -> (proc, shard_idx)
    shard_queue = list(shards_to_build)
    completed = 0
    failed = 0

    def launch_worker(gpu_id, shard_idx, shard_batch_files):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            "python", "-u", worker_script,
            "--gpu_id", str(gpu_id),
            "--shard_idx", str(shard_idx),
            "--batch_files", ",".join(shard_batch_files),
            "--embeddings_dir", embeddings_dir,
            "--shards_dir", shards_dir,
            "--nbits", str(nbits),
            "--kmeans_niters", str(kmeans_niters),
        ]
        proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        return proc

    # Start initial workers
    for gpu_id in range(min(num_gpus, len(shard_queue))):
        shard_idx, shard_batch_files = shard_queue.pop(0)
        proc = launch_worker(gpu_id, shard_idx, shard_batch_files)
        active_procs[gpu_id] = (proc, shard_idx)
        time.sleep(1)

    # Poll and manage workers
    while active_procs or shard_queue:
        time.sleep(5)
        for gpu_id in list(active_procs.keys()):
            proc, shard_idx = active_procs[gpu_id]
            ret = proc.poll()
            if ret is not None:
                if ret == 0:
                    completed += 1
                    print(f"[Progress] {completed}/{len(shards_to_build)} shards complete")
                else:
                    failed += 1
                    print(f"[ERROR] Shard {shard_idx} failed on GPU {gpu_id} (exit code {ret})")
                del active_procs[gpu_id]

                if shard_queue:
                    shard_idx, shard_batch_files = shard_queue.pop(0)
                    proc = launch_worker(gpu_id, shard_idx, shard_batch_files)
                    active_procs[gpu_id] = (proc, shard_idx)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PARALLEL BUILD COMPLETE")
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Shards location: {shards_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Build sharded PLAID index from saved embeddings")

    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing saved embedding batches')
    parser.add_argument('--shards_dir', type=str, required=True,
                        help='Directory to store the sharded indexes')
    parser.add_argument('--batches_per_shard', type=int, default=3,
                        help='Number of embedding batches per shard')
    parser.add_argument('--start_shard', type=int, default=0,
                        help='Shard number to start from (for resuming)')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing shards directory and start fresh')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for indexing (faster but more memory)')
    parser.add_argument('--nbits', type=int, default=4,
                        help='Bits for quantization (2=fastest, 4=balanced, 8=highest quality)')
    parser.add_argument('--kmeans_niters', type=int, default=4,
                        help='K-means iterations (2=fast, 4=balanced, 10+=high quality)')
    parser.add_argument('--parallel', action='store_true',
                        help='Build shards in parallel across multiple GPUs')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use for parallel build (default: all available)')

    args = parser.parse_args()

    total_start = time.time()

    # Handle existing shards directory
    if args.fresh and os.path.exists(args.shards_dir):
        import shutil
        print(f"Removing existing shards at {args.shards_dir}")
        shutil.rmtree(args.shards_dir)

    # Get batch files
    batch_files = sorted([f for f in os.listdir(args.embeddings_dir) if f.endswith('.npz')])
    print(f"Found {len(batch_files)} embedding batches in {args.embeddings_dir}")

    # Calculate number of shards
    num_shards = (len(batch_files) + args.batches_per_shard - 1) // args.batches_per_shard
    print(f"Will create {num_shards} shards ({args.batches_per_shard} batches each)")

    # Create shards directory
    os.makedirs(args.shards_dir, exist_ok=True)

    # Parallel mode
    if args.parallel and args.use_gpu:
        num_gpus = args.num_gpus or torch.cuda.device_count()
        if num_gpus < 1:
            print("No GPUs available for parallel build, falling back to sequential")
        else:
            print(f"Using {num_gpus} GPUs for parallel build")
            build_shards_parallel(
                embeddings_dir=args.embeddings_dir,
                shards_dir=args.shards_dir,
                batch_files=batch_files,
                batches_per_shard=args.batches_per_shard,
                num_gpus=num_gpus,
                nbits=args.nbits,
                kmeans_niters=args.kmeans_niters,
                start_shard=args.start_shard,
            )
            return

    # Sequential mode (original behavior)
    # Device selection
    if args.use_gpu:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            device = [f"cuda:{i}" for i in range(num_gpus)]
            print(f"Using {num_gpus} GPUs: {device}")
        elif num_gpus == 1:
            device = "cuda:0"
            print("Using single GPU: cuda:0")
        else:
            device = "cpu"
            print("No GPU available, falling back to CPU")
    else:
        device = "cpu"
        print("Using CPU for indexing")

    # Build each shard
    total_docs_added = 0
    for shard_idx in range(args.start_shard, num_shards):
        shard_name = f"shard_{shard_idx:04d}"
        shard_path = os.path.join(args.shards_dir, shard_name)

        # Skip if already exists
        if os.path.exists(shard_path):
            print(f"Shard {shard_idx} already exists, skipping")
            continue

        # Get batch files for this shard
        start_batch = shard_idx * args.batches_per_shard
        end_batch = min(start_batch + args.batches_per_shard, len(batch_files))
        shard_batch_files = batch_files[start_batch:end_batch]

        print(f"\n{'='*50}")
        print(f"Building shard {shard_idx}/{num_shards-1}: batches {start_batch}-{end_batch-1}")
        print(f"{'='*50}")
        start = time.time()

        # Load and combine all embeddings for this shard
        all_doc_ids = []
        all_embeddings = []

        for batch_file in tqdm(shard_batch_files, desc=f"Loading batches"):
            batch_path = os.path.join(args.embeddings_dir, batch_file)
            data = np.load(batch_path, allow_pickle=True)
            all_doc_ids.extend(data['ids'].tolist())
            all_embeddings.extend(data['embeddings'].tolist())
            del data
            gc.collect()

        print(f"Loaded {len(all_doc_ids):,} documents for shard {shard_idx}")
        total_docs_added += len(all_doc_ids)

        # Create fresh index for this shard
        print(f"Building PLAID index...")
        print(f"  nbits={args.nbits}, kmeans_niters={args.kmeans_niters}")

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
    print(f"Total documents: {total_docs_added:,}")
    print(f"Number of shards: {num_shards}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Shards location: {args.shards_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
