import json
import logging
import os
from pathlib import Path
from datasets import load_dataset
import sys

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CORPUS_FILE = DATA_DIR / "msmarco_corpus.jsonl"
QUERIES_FILE = DATA_DIR / "msmarco_queries.jsonl"

def ingest_dataset(max_passages: int = 1000):
    logger.info(f"Starting ingestion of MSMARCO-XI dataset (target: {max_passages} unique passages)...")
    
    # We use streaming=True to avoid OOM on large datasets
    # Switching to 'validation' split because 'train' parquets are 1.3GB+ and drop connections
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
            
            # Save query for evaluation later
            queries_data.append({
                "query_id": query_id,
                "query": query
            })

            # Extract passages based on ai4bharat schema
            passages_dict = item.get("passages", {})
            if isinstance(passages_dict, dict):
                p_list = passages_dict.get("Translated_passages", [])
                is_selected = passages_dict.get("is_selected", [])
                
                if not p_list:
                    p_list = passages_dict.get("English_passages", [])
                    
                for idx, text in enumerate(p_list):
                    if not text: continue
                    text = text.strip()
                    if len(text) > 15 and text not in unique_passages:
                        unique_passages.add(text)
                        
                        selected_val = 0
                        if idx < len(is_selected):
                            selected_val = is_selected[idx]
                            
                        passages_data.append({
                            "passage_id": f"p_{len(unique_passages)}",
                            "text": text,
                            "is_selected": selected_val,
                            "source_query_id": query_id
                        })
                        
            if i % 100 == 0:
                logger.info(f"Processed {i} queries, extracted {len(unique_passages)} passages...")
                
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        logger.warning("Dataset streaming failed or was interrupted. We will proceed with whatever data was collected.")

    logger.info(f"Writing {len(passages_data)} passages to {CORPUS_FILE}")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for p in passages_data:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    logger.info(f"Writing {len(queries_data)} queries to {QUERIES_FILE}")
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        for q in queries_data:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # If HF token is available in env, datasets will automatically use it.
    ingest_dataset(max_passages=1000)
