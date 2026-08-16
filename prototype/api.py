import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import warnings

# Suppress HuggingFace and GenAI warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Setup environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global states for our "in memory" vector db
app_state = {
    "model": None,
    "passages": [],
    "passage_vectors": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # COLD START: Everything here runs exactly once when the server boots
    print("\n[Cold Start] Initializing Server...")
    t_start = time.perf_counter()
    
    # 1. Load Embedding Model
    print("[Cold Start] Loading SentenceTransformer...")
    app_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Load Dataset
    print("[Cold Start] Loading Passages...")
    passages = [
        "New Delhi is the capital of India and a major cultural hub.",
        "The Taj Mahal is located in Agra, Uttar Pradesh, India.",
        "Costco, formally known as Costco Wholesale Corp., is a retail store that sells discounted items in bulk, and requires a membership to shop.",
        "Costco does, however, sell wine, as well as a great many other products.",
        "Cricket is the most popular sport in India.",
        "India gained independence on August 15, 1947.",
    ]
    app_state["passages"] = passages
    
    # 3. Pre-embed Passages
    print("[Cold Start] Encoding passages into Vector DB...")
    app_state["passage_vectors"] = app_state["model"].encode(app_state["passages"])
    
    t_end = time.perf_counter()
    print(f"\n[Cold Start] Complete in {(t_end - t_start)*1000:.0f}ms!")
    print("--- SERVER IS HOT AND READY TO RECEIVE QUERIES ---")
    
    yield
    # Cleanup runs on shutdown
    print("Shutting down server...")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    match: str
    latency_ms: float
    breakdown: dict

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ServerError)
)
def get_llm_answer(prompt: str):
    chat = client.chats.create(model='gemini-3.5-flash')
    return chat.send_message(prompt).text.strip()

@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    t_start = time.perf_counter()
    
    # 1. Embed Query
    query_vector = app_state["model"].encode([req.query])[0]
    t_embed = time.perf_counter()
    
    # 2. Search
    similarities = np.dot(app_state["passage_vectors"], query_vector) / (
        np.linalg.norm(app_state["passage_vectors"], axis=1) * np.linalg.norm(query_vector)
    )
    best_idx = np.argmax(similarities)
    context = app_state["passages"][best_idx]
    t_search = time.perf_counter()
    
    # 3. Generate
    prompt = f"Use only this context to answer:\nContext: {context}\n\nQuestion: {req.query}\nAnswer:\n"
    answer = get_llm_answer(prompt)
    t_generate = time.perf_counter()
    
    return QueryResponse(
        answer=answer,
        match=context,
        latency_ms=round((t_generate - t_start) * 1000, 2),
        breakdown={
            "embed_ms": round((t_embed - t_start) * 1000, 2),
            "search_ms": round((t_search - t_embed) * 1000, 2),
            "llm_ms": round((t_generate - t_search) * 1000, 2)
        }
    )
