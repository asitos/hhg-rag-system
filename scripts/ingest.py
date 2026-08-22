import json
import logging
import os
from pathlib import Path
from datasets import load_dataset
import sys

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.chunking.router import ChunkingRouter
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
QUERIES_FILE = DATA_DIR / "msmarco_queries.jsonl"

def ingest_dataset(max_passages: int = 1000):
    logger.info(f"Starting ingestion of MSMARCO-XI dataset (target: {max_passages} unique passages)...")
    
    ds = load_dataset("ai4bharat/MSMARCO-XI", split="validation", streaming=True)
    
    unique_passages = set()
    passages_data = []
    queries_data = []

    try:
        for i, item in enumerate(ds):
            if len(unique_passages) >= max_passages:
                break
                
            query = item.get("query", "")
            query_id = item.get("query_id", str(i))
            queries_data.append({"query_id": query_id, "query": query})

            passages_dict = item.get("passages", {})
            if isinstance(passages_dict, dict):
                p_list = passages_dict.get("Translated_passages", [])
                if not p_list:
                    p_list = passages_dict.get("English_passages", [])
                    
                for idx, text in enumerate(p_list):
                    if not text: continue
                    text = text.strip()
                    if len(text) > 15 and text not in unique_passages:
                        unique_passages.add(text)
                        passages_data.append({
                            "passage_id": f"p_{len(unique_passages)}",
                            "text": text,
                            "language": "hi" if "Translated_passages" in passages_dict and passages_dict.get("Translated_passages") else "en"
                        })
            if i % 100 == 0:
                logger.info(f"Extracted {len(unique_passages)} passages...")
    except Exception as e:
        logger.error(f"Error during streaming: {e}")

    logger.info(f"Writing {len(queries_data)} queries to {QUERIES_FILE}")
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        for q in queries_data:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info("Chunking passages...")
    router = ChunkingRouter()
    all_chunks = []
    for p in passages_data:
        chunks = router.process_passage(p["text"], p["passage_id"], p["language"])
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks. Embedding...")
    
    embedder = Embedder()
    vector_store = VectorStore(persist_dir="data/qdrant_db")
    
    # Process in batches to avoid OOM
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [c.text for c in batch]
        metadatas = [{
            "chunk_id": c.chunk_id,
            "passage_id": c.metadata["passage_id"],
            "text": c.text,
            "strategy": c.strategy,
            "language": c.metadata.get("language", "en")
        } for c in batch]
        
        vectors = embedder.embed_passages(texts)
        # Convert np.ndarray to List[List[float]] for Qdrant client
        vectors_list = vectors.tolist()
        vector_store.add_vectors(vectors_list, metadatas)
        logger.info(f"Upserted batch {i//batch_size + 1}/{(len(all_chunks)+batch_size-1)//batch_size}")

    logger.info("Ingestion complete. Qdrant snapshot saved.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_dataset(max_passages=1000)
