"""
Search across sharded PLAID indexes and merge results.
Supports multi-GPU loading with each shard on a separate GPU.
"""
import os
from typing import List, Dict, Optional, Union
from pylate import indexes, models
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch


class ShardedRetriever:
    """Retriever that searches across multiple PLAID index shards with multi-GPU support."""

    def __init__(
        self,
        shards_dir: str,
        model_name: str = "colbert-ir/colbertv2.0",
        devices: Optional[Union[str, List[str]]] = None,
        model_device: str = "cuda:0",
        use_fp16: bool = False,
    ):
        """
        Initialize the sharded retriever.

        Args:
            shards_dir: Directory containing shard subdirectories
            model_name: ColBERT model to use for encoding queries
            devices: Device(s) for loading shards. Can be:
                - None: Auto-distribute across all available GPUs
                - str: Single device for all shards (e.g., "cuda:0")
                - List[str]: List of devices to distribute shards across
            model_device: Device for the query encoder model
            use_fp16: Use float16 for model to reduce memory
        """
        self.shards_dir = shards_dir
        self.model_device = model_device

        # Load model on specified device
        model_kwargs = {"torch_dtype": torch.float16} if use_fp16 else {}
        print(f"Loading ColBERT model: {model_name} on {model_device}" + (" (fp16)" if use_fp16 else ""))
        self.model = models.ColBERT(model_name_or_path=model_name, model_kwargs=model_kwargs)

        # Get shard directories
        shard_dirs = sorted([
            d for d in os.listdir(shards_dir)
            if d.startswith("shard_") and os.path.isdir(os.path.join(shards_dir, d))
        ])
        num_shards = len(shard_dirs)

        # Determine devices for each shard
        if devices is None:
            # Auto-distribute across all available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                device_list = ["cpu"] * num_shards
            else:
                device_list = [f"cuda:{i % num_gpus}" for i in range(num_shards)]
        elif isinstance(devices, str):
            # Single device for all shards
            device_list = [devices] * num_shards
        else:
            # List of devices - cycle through if fewer devices than shards
            device_list = [devices[i % len(devices)] for i in range(num_shards)]

        self.device_list = device_list

        # Load all shards on their respective devices
        self.shards = []
        print(f"Loading {num_shards} shards across devices: {set(device_list)}")
        for shard_name, device in zip(shard_dirs, device_list):
            print(f"  Loading {shard_name} on {device}")
            shard = indexes.PLAID(
                index_folder=shards_dir,
                index_name=shard_name,
                override=False,
                device=device,
            )
            self.shards.append(shard)
        print(f"Loaded {len(self.shards)} shards")

    def _search_shard(self, shard_idx: int, query_embeddings, k: int):
        """Search a single shard (for parallel execution)."""
        return self.shards[shard_idx](query_embeddings, k=k)

    def search(
        self,
        queries: List[str],
        k: int = 10,
        parallel: bool = True,
    ) -> List[List[Dict]]:
        """
        Search across all shards and return top-k results.

        Args:
            queries: List of query strings
            k: Number of results to return per query
            parallel: Whether to search shards in parallel (default: True)

        Returns:
            List of lists of dicts with 'doc_id' and 'score'
        """
        # Encode queries
        query_embeddings = self.model.encode(
            queries,
            is_query=True,
            batch_size=len(queries),
            show_progress_bar=True,
        )

        # Search each shard
        all_results = []
        if parallel and len(self.shards) > 1:
            # Parallel search across shards using threads
            with ThreadPoolExecutor(max_workers=len(self.shards)) as executor:
                futures = [
                    executor.submit(self._search_shard, i, query_embeddings, k)
                    for i in range(len(self.shards))
                ]
                all_results = [f.result() for f in futures]
        else:
            # Sequential search
            for shard in self.shards:
                shard_results = shard(query_embeddings, k=k)
                all_results.append(shard_results)

        # Merge results across shards
        merged_results = []
        for q_idx in range(len(queries)):
            # Collect results from all shards for this query
            candidates = []
            for shard_results in all_results:
                for result in shard_results[q_idx]:
                    # Handle both dict and object formats
                    if isinstance(result, dict):
                        candidates.append({
                            'doc_id': result.get('id') or result.get('document_id'),
                            'score': result.get('score', 0),
                        })
                    else:
                        candidates.append({
                            'doc_id': result.document_id,
                            'score': result.score,
                        })

            # Sort by score and take top-k
            candidates.sort(key=lambda x: x['score'], reverse=True)
            merged_results.append(candidates[:k])

        return merged_results


def main():
    """Test the sharded retriever."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--shards_dir', type=str, required=True)
    parser.add_argument('--query', type=str, default="What is the capital of France?")
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--devices', type=str, nargs='*', default=None,
                        help='GPU devices for shards (e.g., cuda:0 cuda:1). Default: auto-distribute')
    parser.add_argument('--model_device', type=str, default='cuda:0',
                        help='Device for the query encoder model')
    parser.add_argument('--no_parallel', action='store_true',
                        help='Disable parallel search across shards')
    args = parser.parse_args()

    retriever = ShardedRetriever(
        args.shards_dir,
        devices=args.devices,
        model_device=args.model_device,
    )
    results = retriever.search([args.query], k=args.k, parallel=not args.no_parallel)

    print(f"\nQuery: {args.query}")
    print(f"Top {args.k} results:")
    for i, result in enumerate(results[0]):
        print(f"  {i+1}. {result['doc_id']} (score: {result['score']:.4f})")


if __name__ == "__main__":
    main()
