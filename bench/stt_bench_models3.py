import asyncio
import time
import httpx
from src.config import settings

def get_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

async def benchmark_model(model: str):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": settings.sarvam_api_key}
    audio_bytes = get_audio_bytes("./.venv/lib/python3.14/site-packages/gradio/media_assets/audio/audio_sample.wav")
    
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(url, headers=headers, files={"file": ("a.wav", audio_bytes, "audio/wav")}, data={"model": model, "language_code": "en-IN"})
        except Exception:
            pass
            
        lats = []
        for i in range(5):
            t0 = time.perf_counter()
            resp = await client.post(url, headers=headers, files={"file": ("a.wav", audio_bytes, "audio/wav")}, data={"model": model, "language_code": "en-IN"})
            lats.append((time.perf_counter() - t0)*1000)
            if resp.status_code != 200:
                print(f"{model} failed: {resp.text}")
                return
        print(f"{model} P50: {sorted(lats)[2]:.2f} ms, transcript: {resp.json().get('transcript')}")

async def run_all():
    await benchmark_model("saarika:v2.5")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_all())
