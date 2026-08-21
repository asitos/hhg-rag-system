from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import tempfile
import os

from src.pipeline.harness import PipelineResponse
from src.benchmark.run_benchmarks import build_system
from src.stt.sarvam import SarvamSTT

# Initialize system globally
pipeline, _ = build_system()
stt_client = SarvamSTT()

app = FastAPI(title="HH Goa 2026 - Voice RAG")

class TextQueryRequest(BaseModel):
    query: str

@app.post("/api/query/text", response_model=PipelineResponse)
async def query_text(request: TextQueryRequest):
    try:
        response = pipeline.execute(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/voice")
async def query_voice(audio_file: UploadFile = File(...)):
    """Accepts an audio file, runs STT, then processes via RAG pipeline."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        transcript, stt_latency = stt_client.transcribe(tmp_path)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
            
        response = pipeline.execute(transcript)
        # Inject STT latency into response
        response.stage_latencies_ms["stt"] = stt_latency
        response.total_latency_ms += stt_latency
        
        return response
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
