import os
from dotenv import load_dotenv
load_dotenv()

import json
import time
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.chunking.router import ChunkingRouter
from src.pipeline.harness import RAGPipeline
from src.models import PipelineInput

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_FILE = DATA_DIR / "msmarco_corpus.jsonl"
QUERIES_FILE = DATA_DIR / "msmarco_queries.jsonl"
BENCH_DIR = Path(__file__).resolve().parent

def load_data():
    passages = []
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))
                
    queries = []
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                queries.append(json.loads(line))
                
    return passages, queries

def build_system():
    print("Loading data...")
    passages, queries = load_data()
    
    passages = passages[:500]
    
    print("Chunking data...")
    router = ChunkingRouter()
    all_chunks = []
    for p in passages:
        chunks = router.process_passage(p["text"], p["passage_id"], "en")
        all_chunks.extend(chunks)
        
    print(f"Created {len(all_chunks)} chunks. Embedding...")
    
    metadatas = []
    texts_to_embed = []
    for c in all_chunks:
        texts_to_embed.append(c.text)
        metadatas.append({
            "chunk_id": c.chunk_id,
            "passage_id": c.metadata["passage_id"],
            "text": c.text,
            "strategy": c.strategy,
            "language": c.metadata.get("language", "en")
        })
        
    embedder = Embedder()
    vectors = embedder.embed_passages(texts_to_embed)
    
    print("Building Qdrant index...")
    vector_store = VectorStore()
    vector_store.add_vectors(vectors, metadatas)
    
    print("System built!")
    pipeline = RAGPipeline(vector_store)
    pipeline.embedder = embedder 
    
    return pipeline, queries

def run_benchmarks():
    pipeline, raw_queries = build_system()
    
    # 100 queries
    test_queries = [q["query"] for q in raw_queries[:100]]
    if not test_queries:
        test_queries = ["What is the capital of India?"] * 100
        
    print(f"Running benchmark on {len(test_queries)} queries...")
    
    # Warmup
    pipeline.execute(PipelineInput(query=test_queries[0]))
    
    results = []
    
    for i, q in enumerate(test_queries):
        try:
            res = pipeline.execute(PipelineInput(query=q))
            lat = res.latency
            results.append({
                "query": q,
                "embedding_ms": lat.get("embedding_ms", 0),
                "retrieval_ms": lat.get("retrieval_ms", 0),
                "rerank_ms": lat.get("rerank_ms", 0),
                "generation_ms": lat.get("generation_ms", 0),
                "post_stt_ms": lat.get("post_stt_ms", 0),
                "status": res.guardrail.value
            })
            # Sleep to avoid Gemini free tier rate limit 15 RPM
            time.sleep(4)
        except Exception as e:
            print(f"Query failed: {e}")
            
    # Export CSV
    csv_path = BENCH_DIR / "results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    post_stt = [r["post_stt_ms"] for r in results]
    
    if post_stt:
        print("\n" + "="*40)
        print("⏱️  LATENCY BENCHMARK REPORT")
        print("="*40)
        print(f"P50 Post-STT:  {np.percentile(post_stt, 50):.2f} ms")
        print(f"P70 Post-STT:  {np.percentile(post_stt, 70):.2f} ms")
        print(f"P100 Post-STT: {np.max(post_stt):.2f} ms")
        print(f"Data saved to {csv_path}")
    
if __name__ == "__main__":
    run_benchmarks()
