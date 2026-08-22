import asyncio
import time
import csv
import numpy as np
import sys
from pathlib import Path

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.pipeline.harness import RAGPipeline
from src.models import PipelineInput

def get_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

async def benchmark_full_pipeline(num_requests: int = 10):
    print(f"Starting FULL PIPELINE latency benchmark for {num_requests} requests...")
    
    # Initialize VectorStore from disk
    vector_store = VectorStore(persist_dir="data/qdrant_db")
    count = vector_store.client.count(vector_store.collection_name).count
    print(f"VectorStore initialized with {count} chunks.")
    
    # Initialize Pipeline
    pipeline = RAGPipeline(vector_store)
    audio_path = "./.venv/lib/python3.14/site-packages/gradio/media_assets/audio/audio_sample.wav"
    audio_bytes = get_audio_bytes(audio_path)
    
    print("Running warmup requests...")
    for _ in range(2):
        await pipeline.run(audio_bytes)
        
    results = []
    
    print("Running benchmark...")
    for i in range(num_requests):
        t0 = time.perf_counter()
        resp = await pipeline.run(audio_bytes)
        full_lat = (time.perf_counter() - t0) * 1000
        
        # resp.latency dictionary contains individual stage timings
        lat = resp.latency
        results.append({
            "stt_ms": lat.get("stt_ms", 0),
            "guardrail_pre_ms": lat.get("guardrail_pre_ms", 0),
            "embedding_ms": lat.get("embedding_ms", 0),
            "retrieval_ms": lat.get("retrieval_ms", 0),
            "rerank_ms": lat.get("rerank_ms", 0),
            "generation_ms": lat.get("generation_ms", 0),
            "guardrail_post_ms": lat.get("guardrail_post_ms", 0),
            "post_stt_ms": lat.get("post_stt_ms", 0),
            "full_pipeline_ms": full_lat
        })
        print(f"Req {i+1}: STT {lat.get('stt_ms',0):.1f}ms, Post-STT {lat.get('post_stt_ms',0):.1f}ms -> Full {full_lat:.1f}ms")
        
    print("\n--- LATENCY PERCENTILES ---")
    keys = ["stt_ms", "post_stt_ms", "full_pipeline_ms", "embedding_ms", "retrieval_ms", "rerank_ms", "generation_ms"]
    for k in keys:
        vals = [r[k] for r in results]
        print(f"\n{k.upper()}:")
        print(f"P50:  {np.percentile(vals, 50):.2f} ms")
        print(f"P95:  {np.percentile(vals, 95):.2f} ms")
        print(f"P100: {np.max(vals):.2f} ms")
        
    with open("bench/latency_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    await pipeline.stt.close()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(benchmark_full_pipeline(10))
