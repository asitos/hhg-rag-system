import re

with open("app.py", "r") as f:
    content = f.read()

init_logic = """
from src.retrieval.embedder import Embedder
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model Loading and Initialization
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
"""

# Replace ONLY the lifespan function non-greedily
content = re.sub(r'@asynccontextmanager\nasync def lifespan\(app: FastAPI\):.*?    yield', init_logic, content, flags=re.DOTALL | re.MULTILINE)

# Add scenario endpoint
scenario_endpoint = """
from pydantic import BaseModel
class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/v1/scenario")
async def set_scenario(req: ScenarioRequest):
    settings.demo_scenario = req.scenario
    return {"status": "ok", "scenario": settings.demo_scenario}

@app.get("/api/v1/scenario")
async def get_scenario():
    return {"scenario": settings.demo_scenario}
"""
content = content.replace('from fastapi import FastAPI, UploadFile, File, Form, HTTPException', 
                          'from fastapi import FastAPI, UploadFile, File, Form, HTTPException\n' + scenario_endpoint)

with open("app.py", "w") as f:
    f.write(content)
