from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
import os
import io
from pydub import AudioSegment

from src.pipeline.harness import RAGPipeline, PipelineInput
from src.retrieval.vector_store import VectorStore
from src.stt.sarvam import SarvamSTT
from src.stt.sarvam_stream import SarvamStreamSession
from src.config import settings

from contextlib import asynccontextmanager


from src.retrieval.embedder import Embedder
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Vector Store...")
    vs = VectorStore(persist_dir="data/qdrant_db")
    
    if settings.app_mode in ["mock", "demo"]:
        count = vs.client.count(vs.collection_name).count
        if count == 0:
            print("WARNING: Qdrant DB is empty. Building demo index now...")
            embedder = Embedder(settings.embedding_model_id)
            with open("tests/fixtures/passages.json", "r") as f:
                passages = json.load(f)
            vecs = []
            payloads = []
            for p in passages:
                vecs.append(embedder.embed_query(p["text"]))
                payloads.append({
                    "chunk_id": p["id"],
                    "text": p["text"],
                    "language": p["language"],
                    "strategy": "semantic"
                })
            vs.add_vectors(vecs, payloads)
            print(f"Added {len(vecs)} vectors to Qdrant demo index.")

    global pipeline
    print("Loading RAG Models...")
    pipeline = RAGPipeline(vs)
    yield
    if hasattr(rest_stt, "close"):
        await rest_stt.close()


app = FastAPI(title="HH Goa 2026 Voice RAG API", lifespan=lifespan)

# Setup CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]
if settings.frontend_origin:
    origins.append(settings.frontend_origin)


class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/v1/scenario")
async def set_scenario(req: ScenarioRequest):
    settings.demo_scenario = req.scenario
    return {"status": "ok", "scenario": settings.demo_scenario}

@app.get("/api/v1/scenario")
async def get_scenario():
    return {"scenario": settings.demo_scenario}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore()
pipeline = RAGPipeline(vector_store)
if settings.app_mode == "mock":
    from src.stt.mock import MockSTT, MockStreamSession
    rest_stt = MockSTT()
else:
    rest_stt = SarvamSTT()

from contextlib import asynccontextmanager



class LatencyMetrics(BaseModel):
    stt_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

class VoiceResponse(BaseModel):
    answer: str
    transcript: str
    sources: List[Dict[str, str]]
    guardrail: str
    guardrail_reason: Optional[str] = None
    latency: LatencyMetrics

@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": settings.app_mode}

@app.get("/api/v1/health")
async def api_health_check():
    return {"status": "ok", "mode": settings.app_mode}

@app.post("/api/v1/text", response_model=VoiceResponse)
async def process_text_query(query: str = Form(...)):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        p_input = PipelineInput(query=query)
        response = await pipeline.execute(p_input)
        
        sources = [{"chunk_id": s.chunk_id, "text": s.text, "score": str(s.score)} for s in response.sources]
        
        lat = LatencyMetrics(
            stt_ms=0.0,
            embedding_ms=response.latency.get("embedding_ms", 0),
            retrieval_ms=response.latency.get("retrieval_ms", 0),
            rerank_ms=response.latency.get("reranking_ms", 0),
            generation_ms=response.latency.get("generation_ms", 0),
            total_ms=response.latency.get("post_stt_total_ms", 0)
        )
        
        return VoiceResponse(
            answer=response.answer,
            transcript=query,
            sources=sources,
            guardrail=response.guardrail.value,
            guardrail_reason=response.guardrail_reason,
            latency=lat
        )
    except Exception as e:
        import logging
        logging.error(f"Text query error: {str(e)}", exc_info=True)
        error_str = str(e)
        if hasattr(e, 'last_attempt') and e.last_attempt is not None:
            try:
                e.last_attempt.result()
            except Exception as inner_e:
                error_str = str(inner_e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "ClientError" in str(e):
            raise HTTPException(status_code=429, detail="Gemini API daily quota (20 requests/day) exceeded. Ensure you have enabled GCP billing on your AI Studio project.")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/voice", response_model=VoiceResponse)
async def process_voice_query(audio: UploadFile = File(...)):
    t0 = time.perf_counter()
    audio_bytes = await audio.read()
    
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
        
    try:
        # Convert incoming WebM/OGG to WAV for Sarvam using pydub
        import tempfile
        from pydub import AudioSegment
        import io
        
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()

        # Transcribe
        transcript = await rest_stt.transcribe(wav_bytes)
        
        stt_ms = (time.perf_counter() - t0) * 1000
        
        if not transcript.strip():
            # If no transcript, return early
            lat = LatencyMetrics(stt_ms=stt_ms, total_ms=stt_ms)
            return VoiceResponse(
                answer="Could not understand audio.",
                transcript="",
                sources=[],
                guardrail="PASS",
                latency=lat
            )
            
        p_input = PipelineInput(query=transcript)
        response = await pipeline.execute(p_input)
        
        sources = [{"chunk_id": s.chunk_id, "text": s.text, "score": str(s.score)} for s in response.sources]
        
        lat = LatencyMetrics(
            stt_ms=stt_ms,
            embedding_ms=response.latency.get("embedding_ms", 0),
            retrieval_ms=response.latency.get("retrieval_ms", 0),
            rerank_ms=response.latency.get("reranking_ms", 0),
            generation_ms=response.latency.get("generation_ms", 0),
            total_ms=stt_ms + response.latency.get("post_stt_total_ms", 0)
        )
        
        return VoiceResponse(
            answer=response.answer,
            transcript=transcript,
            sources=sources,
            guardrail=response.guardrail.value,
            guardrail_reason=response.guardrail_reason,
            latency=lat
        )
    except Exception as e:
        import logging
        import traceback
        logging.error(f"Voice query error: {str(e)}", exc_info=True)
        
        # Unwrap tenacity.RetryError if present to catch the underlying 429
        error_str = str(e)
        if hasattr(e, 'last_attempt') and e.last_attempt is not None:
            try:
                e.last_attempt.result()
            except Exception as inner_e:
                error_str = str(inner_e)
                
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "ClientError" in str(e):
            raise HTTPException(status_code=429, detail="Gemini API daily quota (20 requests/day) exceeded. Ensure you have enabled GCP billing on your AI Studio project.")
            
        raise HTTPException(status_code=500, detail="Internal server error")


from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/api/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if settings.app_mode == "mock":
        session = MockStreamSession()
    else:
        session = SarvamStreamSession()
    success = await session.connect()
    
    if not success:
        await websocket.close(code=1011, reason="Failed to connect to Sarvam")
        return
        
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                # Expecting raw 16kHz mono PCM bytes
                import numpy as np
                import scipy.signal
                y = np.frombuffer(data["bytes"], dtype=np.int16)
                await session.send_chunk((16000, y))
                
                # Send back any partials
                if session.latest_partial:
                    await websocket.send_json({"type": "partial", "text": session.latest_partial})
                    
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "stop":
                    transcript, metrics = await session.finalize()
                    if not transcript.strip():
                        await websocket.send_json({"type": "error", "message": "No speech detected"})
                        break
                        
                    # Run RAG
                    p_input = PipelineInput(query=transcript)
                    response = await pipeline.execute(p_input)
                    
                    sources = [{"chunk_id": s.chunk_id, "text": s.text, "score": str(s.score)} for s in response.sources]
                    
                    stt_ms = metrics.get("stt_final_ms", 0)
                    lat = {
                        "stt_ms": stt_ms,
                        "embedding_ms": response.latency.get("embedding_ms", 0),
                        "retrieval_ms": response.latency.get("retrieval_ms", 0),
                        "rerank_ms": response.latency.get("reranking_ms", 0),
                        "generation_ms": response.latency.get("generation_ms", 0),
                        "total_ms": stt_ms + response.latency.get("post_stt_total_ms", 0)
                    }
                    
                    await websocket.send_json({
                        "type": "final",
                        "answer": response.answer,
                        "transcript": transcript,
                        "sources": sources,
                        "guardrail": response.guardrail.value,
                        "latency": lat
                    })
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import logging
        logging.error(f"WS error: {e}", exc_info=True)
    finally:
        await session.close()


import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve the static frontend locally as a convenience
if os.path.isdir("frontend"):
    @app.get("/")
    async def serve_index():
        return FileResponse("frontend/index.html")
    app.mount("/", StaticFiles(directory="frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
