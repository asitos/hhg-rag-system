import asyncio
import time
import csv
import numpy as np
import os
import wave
from pathlib import Path
import sys

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stt.sarvam import SarvamSTT

def get_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

async def benchmark_stt(num_requests: int = 20):
    print(f"Starting STT benchmark for {num_requests} requests...")
    
    stt = SarvamSTT()
    audio_path = "./.venv/lib/python3.14/site-packages/gradio/media_assets/audio/audio_sample.wav"
    audio_bytes = get_audio_bytes(audio_path)
    
    results = []
    
    # Warmup
    print("Running warmup request...")
    await stt.transcribe(audio_bytes, language="en-IN")
    
    for i in range(num_requests):
        t0 = time.perf_counter()
        transcript = await stt.transcribe(audio_bytes, language="en-IN")
        latency = (time.perf_counter() - t0) * 1000
        
        results.append({
            "request_num": i + 1,
            "latency_ms": latency,
            "transcript_len": len(transcript)
        })
        print(f"Request {i+1}: {latency:.2f} ms")
        
    latencies = [r["latency_ms"] for r in results]
    
    print("\n--- BASELINE STT LATENCY ---")
    print(f"P50:  {np.percentile(latencies, 50):.2f} ms")
    print(f"P70:  {np.percentile(latencies, 70):.2f} ms")
    print(f"P95:  {np.percentile(latencies, 95):.2f} ms")
    print(f"P99:  {np.percentile(latencies, 99):.2f} ms")
    print(f"P100: {np.max(latencies):.2f} ms")
    
    with open("bench/stt_baseline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(benchmark_stt(20))
