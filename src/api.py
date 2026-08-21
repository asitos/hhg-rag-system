from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

from src.pipeline.harness import RAGPipeline
from bench.bench import build_system
from src.stt.sarvam import SarvamSTT
from src.models import PipelineInput, PipelineOutput

# Initialize system globally
try:
    pipeline, _ = build_system()
except Exception as e:
    pipeline = None
    print(f"Warning: Could not initialize pipeline. Ensure index exists. Error: {e}")

stt_client = SarvamSTT()

app = FastAPI(title="HH Goa 2026 - Voice RAG")

@app.post("/api/query/text", response_model=PipelineOutput)
async def query_text(request: PipelineInput):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        response = pipeline.execute(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/voice", response_model=PipelineOutput)
async def query_voice(audio_file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        transcript, stt_latency = stt_client.transcribe(tmp_path)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
            
        p_input = PipelineInput(query=transcript)
        response = pipeline.execute(p_input)
        
        # Inject STT latency into response
        response.latency["stt_ms"] = stt_latency
        response.latency["total_ms"] = response.latency["post_stt_ms"] + stt_latency
        
        return response
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
