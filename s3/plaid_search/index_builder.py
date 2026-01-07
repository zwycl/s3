import json
import argparse
import time
import math
import os
import torch
from tqdm import tqdm
from pylate import models
import numpy as np


def count_lines(filepath):
    """Count lines in file for progress bar total."""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def get_visible_gpus():
    """Get list of visible CUDA devices."""
    if not torch.cuda.is_available():
        return []

    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        # Return the number of devices specified
        devices = [d.strip() for d in cuda_visible.split(',') if d.strip()]
        return list(range(len(devices)))
    else:
        # Return all available devices
        return list(range(torch.cuda.device_count()))


def encode_batch_with_progress(model, sentences, pool, batch_size, is_query=False, chunk_size=None, pool_factor=1):
    """Encode a batch with multi-GPU and show progress bar."""
    if chunk_size is None:
        chunk_size = min(
            math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000
        )

    num_chunks = math.ceil(len(sentences) / chunk_size)

    input_queue = pool["input"]
    last_chunk_id = 0
    chunk = []

    # Send chunks to workers
    for sentence in sentences:
        chunk.append(sentence)
        if len(chunk) >= chunk_size:
            input_queue.put([
                last_chunk_id, batch_size, chunk, None, None,
                "float32", True, False, is_query, pool_factor, 1,
            ])
            last_chunk_id += 1
            chunk = []

    if len(chunk) > 0:
        input_queue.put([
            last_chunk_id, batch_size, chunk, None, None,
            "float32", True, False, is_query, pool_factor, 1,
        ])
        last_chunk_id += 1

    # Collect results with progress bar
    output_queue = pool["output"]
    results_list = []
    for i in tqdm(range(last_chunk_id), desc="Encoding chunks", unit="chunk"):
        try:
            result = output_queue.get(timeout=300)  # 5 min timeout per chunk
            results_list.append(result)
        except Exception as e:
            print(f"\n  ERROR: Timeout waiting for chunk {i}. Workers may have failed.")
            raise e

    results_list = sorted(results_list, key=lambda x: x[0])

    # Concatenate results
    embeddings = []
    for result in results_list:
        embeddings.extend(result[1])

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Build ColBERT embeddings from corpus using all visible GPUs")

    parser.add_argument('--corpus_path', type=str, required=True,
                        help='Path to corpus file (jsonl format with id and contents fields)')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--checkpoint', type=str, default='colbert-ir/colbertv2.0',
                        help='ColBERT model checkpoint')
    parser.add_argument('--max_document_length', type=int, default=256,
                        help='Maximum document length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding documents')
    parser.add_argument('--mega_batch_size', type=int, default=500000,
                        help='Number of documents to process before saving to disk')
    parser.add_argument('--pool_factor', type=int, default=1,
                        help='Pool neighboring token embeddings (2=halves index size, 1=no pooling)')

    args = parser.parse_args()

    total_start = time.time()

    # Create embeddings directory
    os.makedirs(args.embeddings_dir, exist_ok=True)

    # Detect visible GPUs
    visible_gpus = get_visible_gpus()
    if not visible_gpus:
        raise RuntimeError("No CUDA devices available!")

    print(f"Detected {len(visible_gpus)} visible GPU(s): {visible_gpus}")
    device_list = [f"cuda:{i}" for i in visible_gpus]

    # Count total documents first
    print(f"Counting documents in {args.corpus_path}...")
    total_docs = count_lines(args.corpus_path)
    print(f"Found {total_docs:,} documents")

    num_mega_batches = math.ceil(total_docs / args.mega_batch_size)
    print(f"Will process in {num_mega_batches} mega-batches of {args.mega_batch_size:,} docs each")

    # Initialize ColBERT model
    print(f"\n[1/2] Loading ColBERT model: {args.checkpoint}")
    start = time.time()
    model = models.ColBERT(
        model_name_or_path=args.checkpoint,
        document_length=args.max_document_length,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Start multi-process pool with all visible GPUs
    pool = model.start_multi_process_pool(target_devices=device_list)
    print(f"Started {len(pool['processes'])} GPU workers on devices: {device_list}")

    # Process in mega-batches
    print(f"\n[2/2] Encoding documents in mega-batches...")
    start = time.time()

    mega_batch_idx = 0
    current_batch_docs = []
    current_batch_ids = []
    doc_idx = 0

    with open(args.corpus_path, "r") as f:
        for line in f:
            item = json.loads(line)
            current_batch_docs.append(item["contents"])
            current_batch_ids.append(str(item["id"]))
            doc_idx += 1

            # Process mega-batch when full
            if len(current_batch_docs) >= args.mega_batch_size:
                print(f"\nEncoding mega-batch {mega_batch_idx + 1}/{num_mega_batches} ({len(current_batch_docs):,} docs, {doc_idx:,}/{total_docs:,} total)...")

                # Encode
                embeddings = encode_batch_with_progress(
                    model=model,
                    sentences=current_batch_docs,
                    pool=pool,
                    batch_size=args.batch_size,
                    is_query=False,
                    pool_factor=args.pool_factor,
                )

                # Save to disk
                batch_file = os.path.join(args.embeddings_dir, f"batch_{mega_batch_idx:04d}.npz")
                np.savez(
                    batch_file,
                    ids=np.array(current_batch_ids, dtype=object),
                    embeddings=np.array(embeddings, dtype=object),
                )
                print(f"  Saved to {batch_file}")

                # Clear memory
                del embeddings
                current_batch_docs = []
                current_batch_ids = []
                mega_batch_idx += 1

    # Process remaining documents
    if current_batch_docs:
        print(f"\nEncoding mega-batch {mega_batch_idx + 1}/{num_mega_batches} ({len(current_batch_docs):,} docs, {doc_idx:,}/{total_docs:,} total)...")

        embeddings = encode_batch_with_progress(
            model=model,
            sentences=current_batch_docs,
            pool=pool,
            batch_size=args.batch_size,
            is_query=False,
            pool_factor=args.pool_factor,
        )

        batch_file = os.path.join(args.embeddings_dir, f"batch_{mega_batch_idx:04d}.npz")
        np.savez(
            batch_file,
            ids=np.array(current_batch_ids, dtype=object),
            embeddings=np.array(embeddings, dtype=object),
        )
        print(f"  Saved to {batch_file}")
        del embeddings

    # Stop multi-process pool
    model.stop_multi_process_pool(pool)

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Embeddings built successfully!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Embeddings location: {args.embeddings_dir}")
    print(f"Total documents: {total_docs:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
