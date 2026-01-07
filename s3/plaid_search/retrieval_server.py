import os
from typing import List, Optional
import argparse

import torch
from tqdm import tqdm
import datasets

from sharded_retriever import ShardedRetriever

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


class ShardedPLAIDRetriever:
    """Retriever using ShardedRetriever for multi-GPU parallel search."""

    def __init__(
        self,
        shards_dir: str,
        corpus_path: str,
        model_name: str = "colbert-ir/colbertv2.0",
        topk: int = 10,
        batch_size: int = 32,
        use_fp16: bool = False,
    ):
        self.topk = topk
        self.batch_size = batch_size

        # Get available GPUs
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else None

        # Initialize ShardedRetriever
        print(f"Initializing ShardedRetriever with {num_gpus} GPUs" + (" (fp16)" if use_fp16 else ""))
        self.retriever = ShardedRetriever(
            shards_dir=shards_dir,
            model_name=model_name,
            devices=devices,
            model_device="cuda:0" if num_gpus > 0 else "cpu",
            use_fp16=use_fp16,
        )

        # Load corpus for document lookup
        self.corpus = load_corpus(corpus_path)
        print("ShardedPLAID retriever initialized")

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        results, scores = self.batch_search([query], num, return_score=True)
        if return_score:
            return results[0], scores[0]
        return results[0]

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        all_results = []
        all_scores = []

        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Sharded PLAID retrieval'):
            batch_queries = query_list[start_idx:start_idx + self.batch_size]

            # Use ShardedRetriever's parallel search
            search_results = self.retriever.search(batch_queries, k=num, parallel=True)

            for query_results in search_results:
                doc_ids = [int(r['doc_id']) for r in query_results]
                scores = [r['score'] for r in query_results]

                docs = load_docs(self.corpus, doc_ids)

                all_results.append(docs)
                all_scores.append(scores)

        if return_score:
            return all_results, all_scores
        return all_results


#####################################
# FastAPI server below
#####################################

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = retriever.topk

    # Perform batch retrieval
    if request.return_scores:
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=True
        )
    else:
        results = retriever.batch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=False
        )
        scores = None

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores and scores is not None:
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the sharded PLAID retriever server.")
    parser.add_argument("--shards_dir", type=str, required=True, help="Path to sharded PLAID index directory.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=10, help="Number of retrieved passages per query.")
    parser.add_argument("--retriever_model", type=str, default="colbert-ir/colbertv2.0", help="ColBERT model path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for retrieval.")
    parser.add_argument("--use_fp16", action="store_true", help="Use float16 for model to reduce memory.")
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on.')

    args = parser.parse_args()

    print("Initializing Sharded PLAID Retriever...")
    retriever = ShardedPLAIDRetriever(
        shards_dir=args.shards_dir,
        corpus_path=args.corpus_path,
        model_name=args.retriever_model,
        topk=args.topk,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
    )
    print("Retriever initialized")

    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
