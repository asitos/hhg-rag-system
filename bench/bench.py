import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from src.config import settings
import json
import time
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.pipeline.harness import RAGPipeline
from src.models import PipelineInput

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUERIES_FILE = DATA_DIR / "msmarco_queries.jsonl"
BENCH_DIR = Path(__file__).resolve().parent

def load_queries():
    queries = []
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                queries.append(json.loads(line))
    return queries

def build_system():
    print("Loading system components from disk...")
    
    embedder = Embedder()
    # Load Qdrant from disk snapshot instead of re-indexing
    vector_store = VectorStore(persist_dir="data/qdrant_db")
    
    count = vector_store.client.count(vector_store.collection_name).count
    if count == 0:
        raise RuntimeError("Vector database is empty! Please run scripts/ingest.py first.")
        
    print(f"System built! Loaded {count} vectors.")
    pipeline = RAGPipeline(vector_store)
    pipeline.embedder = embedder 
    
    queries = load_queries()
    return pipeline, queries

def run_benchmarks(num_queries=100, output_csv="bench/results.csv"):
    pipeline, raw_queries = build_system()
    
    test_queries = [q["query"] for q in raw_queries[:num_queries]]
    if not test_queries:
        test_queries = ["What is the capital of India?"] * num_queries
        
    print(f"Running benchmark on {len(test_queries)} queries...")
    
    # Setup mock file based on mode
    if settings.app_mode == "mock":
        output_csv = output_csv.replace(".csv", "-mock.csv")
        
    print(f"Running in APP_MODE={settings.app_mode}")
    
    # Warmup
    asyncio.run(pipeline.execute(PipelineInput(query=test_queries[0])))
    
    results = []
    
    for i, q in enumerate(test_queries):
        try:
            res = asyncio.run(pipeline.execute(PipelineInput(query=q)))
            lat = res.latency
            results.append({
                "query": q,
                "embedding_ms": lat.get("embedding_ms", 0),
                "retrieval_ms": lat.get("retrieval_ms", 0),
                "rerank_ms": lat.get("reranking_ms", 0),
                "generation_ms": lat.get("generation_ms", 0),
                "post_stt_ms": lat.get("post_stt_total_ms", 0),
                "status": res.guardrail.value
            })
            time.sleep(4)
        except Exception as e:
            print(f"Query failed: {e}")
            
    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    post_stt = [r["post_stt_total_ms"] for r in results]
    
    if post_stt:
        print("\n" + "="*40)
        print("⏱️  LATENCY BENCHMARK REPORT")
        print("="*40)
        print(f"P50 Post-STT:  {np.percentile(post_stt, 50):.2f} ms")
        print(f"P70 Post-STT:  {np.percentile(post_stt, 70):.2f} ms")
        print(f"P100 Post-STT: {np.max(post_stt):.2f} ms")
        print(f"Data saved to {output_csv}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=str, default="bench/results.csv")
    args = parser.parse_args()
    
    run_benchmarks(args.queries, args.output)
