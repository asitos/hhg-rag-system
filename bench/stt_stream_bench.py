import asyncio
import time
import numpy as np
import scipy.io.wavfile as wavfile
from src.stt.sarvam_stream import SarvamStreamSession
from src.stt.sarvam import SarvamSTT

async def benchmark_streaming():
    with open(".venv/lib/python3.14/site-packages/gradio/media_assets/audio/audio_sample.wav", "rb") as f:
        audio_bytes = f.read()
        
    sr, audio_data = wavfile.read(".venv/lib/python3.14/site-packages/gradio/media_assets/audio/audio_sample.wav")
    
    # 1. REST Benchmark
    rest = SarvamSTT()
    print("--- REST Benchmark ---")
    rest_latencies = []
    for i in range(5):
        t0 = time.perf_counter()
        txt = await rest.transcribe(audio_bytes)
        lat = (time.perf_counter() - t0) * 1000
        rest_latencies.append(lat)
        print(f"REST {i+1}: {lat:.2f} ms")
        
    print(f"REST P50: {np.percentile(rest_latencies, 50):.2f} ms")
    
    # 2. Streaming Benchmark
    print("\n--- Streaming Benchmark ---")
    stream_first_partial = []
    stream_final = []
    
    chunk_samples = int(sr * 0.1) # 100ms
    
    for i in range(5):
        sess = SarvamStreamSession()
        await sess.connect()
        
        # padded to trigger VAD
        padded_audio = np.concatenate([audio_data, np.zeros(int(sr * 0.5), dtype=audio_data.dtype)])
        
        t_physical_end = None
        for j in range(0, len(padded_audio), chunk_samples):
            chunk = padded_audio[j:j+chunk_samples]
            await sess.send_chunk((sr, chunk))
            
            # Record physical speech end when we hit the zero-padding
            if t_physical_end is None and j >= len(audio_data):
                t_physical_end = time.perf_counter()
                
            await asyncio.sleep(0.1) # realtime
            
        trans, metrics = await sess.finalize()
        await sess.close()
        
        # Calculate from physical end for streaming
        if sess.t_final and t_physical_end:
            ttff = (sess.t_final - t_physical_end) * 1000
        else:
            ttff = metrics.get('stt_final_ms', 0)
            
        ttfp = metrics.get('stt_first_partial_ms', 0)
        stream_first_partial.append(ttfp)
        stream_final.append(ttff)
        
        print(f"Stream {i+1}: First Partial: {ttfp:.2f} ms, Final: {ttff:.2f} ms")
        
    print(f"Stream First Partial P50: {np.percentile(stream_first_partial, 50):.2f} ms")
    print(f"Stream Final P50: {np.percentile(stream_final, 50):.2f} ms")
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(benchmark_streaming())
