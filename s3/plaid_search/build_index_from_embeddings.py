import os
import argparse
import time
import gc
from tqdm import tqdm
from pylate import indexes
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Build FastPLAID index from saved embeddings")

    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing saved embedding batches')
    parser.add_argument('--index_name', type=str, default='wiki18-colbert',
                        help='Name for the output index')
    parser.add_argument('--index_folder', type=str, default='/home/ubuntu/s3/data',
                        help='Folder to store the index')
    parser.add_argument('--start_batch', type=int, default=0,
                        help='Batch number to start from (for resuming)')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing index and start fresh')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for indexing (faster but more memory)')
    parser.add_argument('--nbits', type=int, default=4,
                        help='Bits for quantization (2=fastest, 4=balanced, 8=highest quality)')
    parser.add_argument('--kmeans_niters', type=int, default=4,
                        help='K-means iterations (2=fast, 4=balanced, 10+=high quality)')

    args = parser.parse_args()

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

    total_start = time.time()

    # Get batch files
    batch_files = sorted([f for f in os.listdir(args.embeddings_dir) if f.endswith('.npz')])
    print(f"Found {len(batch_files)} embedding batches in {args.embeddings_dir}")

    # Handle existing index
    index_path = os.path.join(args.index_folder, args.index_name)
    if args.fresh and os.path.exists(index_path):
        import shutil
        print(f"Removing existing index at {index_path}")
        shutil.rmtree(index_path)

    # Build/load FastPLAID index
    resuming = os.path.exists(index_path) and args.start_batch > 0
    if resuming:
        print(f"\nResuming FastPLAID index from batch {args.start_batch}: {args.index_name}")
    else:
        print(f"\nBuilding FastPLAID index: {args.index_name}")
        print(f"  nbits={args.nbits}, kmeans_niters={args.kmeans_niters}")
    start = time.time()

    index = indexes.PLAID(
        index_folder=args.index_folder,
        index_name=args.index_name,
        override=not resuming,
        show_progress=True,
        device=device,
        use_fast=True,  # Use FastPLAID backend
        nbits=args.nbits,
        kmeans_niters=args.kmeans_niters,
    )

    # Skip already processed batches
    batch_files = batch_files[args.start_batch:]
    print(f"Processing batches {args.start_batch} to {args.start_batch + len(batch_files) - 1}")

    # Load and add each batch
    total_docs_added = 0
    for i, batch_file in enumerate(tqdm(batch_files, desc="Adding batches to index")):
        batch_path = os.path.join(args.embeddings_dir, batch_file)
        data = np.load(batch_path, allow_pickle=True)
        doc_ids = data['ids'].tolist()
        embeddings = data['embeddings'].tolist()

        actual_batch = args.start_batch + i

        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=embeddings,
        )

        total_docs_added += len(doc_ids)

        # Clear memory after each file
        del data, doc_ids, embeddings
        gc.collect()

        print(f"  Completed batch {actual_batch} ({total_docs_added:,} docs total)")

    print(f"Index built in {time.time() - start:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"FastPLAID index built successfully!")
    print(f"Total documents: {total_docs_added:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Index location: {args.index_folder}/{args.index_name}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
