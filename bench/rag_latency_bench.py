import asyncio
import time
import csv
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.vector_store import VectorStore
from src.pipeline.harness import RAGPipeline
from src.models import PipelineInput

async def benchmark_rag():
    print("Starting RAG benchmark...")
    vector_store = VectorStore(persist_dir="data/qdrant_db")
    pipeline = RAGPipeline(vector_store)
    
    # Warmup
    for _ in range(2):
        pipeline.execute(PipelineInput(query="What is the capital of India?"))
        
    results = []
    
    for i in range(10):
        t0 = time.perf_counter()
        resp = pipeline.execute(PipelineInput(query="What is the capital of India?"))
        full_lat = (time.perf_counter() - t0) * 1000
        lat = resp.latency
        results.append({
            "guardrail_pre_ms": lat.get("guardrail_pre_ms", 0),
            "embedding_ms": lat.get("embedding_ms", 0),
            "retrieval_ms": lat.get("retrieval_ms", 0),
            "rerank_ms": lat.get("rerank_ms", 0),
            "generation_ms": lat.get("generation_ms", 0),
            "guardrail_post_ms": lat.get("guardrail_post_ms", 0),
            "post_stt_ms": lat.get("post_stt_ms", 0)
        })
        print(f"Req {i+1}: {lat.get('post_stt_ms',0):.1f}ms")
        
    for k in results[0].keys():
        vals = [r[k] for r in results]
        print(f"{k.upper()}: P50 {np.percentile(vals, 50):.2f} ms")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(benchmark_rag())
