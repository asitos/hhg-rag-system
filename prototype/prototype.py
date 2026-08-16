import os
import time
import logging

t_start = time.perf_counter()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
from google import genai
import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from google.genai.errors import ServerError

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
t_imports = time.perf_counter()

model = SentenceTransformer("all-MiniLM-L6-v2")
t_model_load = time.perf_counter()

print("Downloading MSMARCO dataset (first 500 passages)...")
from datasets import load_dataset

ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
passages = []
for item in ds:
    if len(passages) >= 500:
        break
    for p in item.get("positive_passages", []):
        passages.append(p["text"])
    for p in item.get("negative_passages", []):
        passages.append(p["text"])

passages = list(set(passages))[:500]

print(f"Loaded {len(passages)} real passages from MSMARCO.")
passage_vectors = model.encode(passages)
t_passage_encode = time.perf_counter()

print("\n--- SERVER READY --- (Cold start complete)\n")

query = "what does costco sell"
t_query_start = time.perf_counter()

query_vector = model.encode([query])[0]
t_query_embed = time.perf_counter()

similarities = np.dot(passage_vectors, query_vector) / (
    np.linalg.norm(passage_vectors, axis=1) * np.linalg.norm(query_vector)
)
best_idx = np.argmax(similarities)
context = passages[best_idx]
print(f"Best match: {context}")
t_search = time.perf_counter()

prompt = f"Use only this context to answer:\nContext: {context}\n\nQuestion: {query}\nAnswer:\n"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ServerError),
)
def get_answer():
    chat = client.chats.create(model="gemini-3.5-flash")
    return chat.send_message(prompt)


response = get_answer()
print(f"Answer: {response.text.strip()}")
t_generate = time.perf_counter()

print("\n=== TIMING REPORT ===")
print(f"[Cold Start] Imports & Setup:      {(t_imports - t_start) * 1000:.0f} ms")
print(f"[Cold Start] Load AI Model to RAM: {(t_model_load - t_imports) * 1000:.0f} ms")
print(
    f"[Cold Start] Encode Passages:      {(t_passage_encode - t_model_load) * 1000:.0f} ms"
)
print("-" * 42)
print(
    f"[Hot Path]   Embed your Question:     {(t_query_embed - t_query_start) * 1000:.0f} ms"
)
print(
    f"[Hot Path]   Vector Database Search:  {(t_search - t_query_embed) * 1000:.0f} ms"
)
print(f"[Hot Path]   Gemini API Network Call: {(t_generate - t_search) * 1000:.0f} ms")
print("-" * 42)
print(
    f"Total Response Time (Hot Path):       {(t_generate - t_query_start) * 1000:.0f} ms"
)
