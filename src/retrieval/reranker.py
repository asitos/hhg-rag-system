from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Loads a fast cross-encoder model for re-ranking candidate chunks."""
        self.model = CrossEncoder(model_id, max_length=512)
        
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        # Cross encoder requires pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        import numpy as np
        scores = self.model.predict(pairs)
        scores = np.atleast_1d(scores)
        
        # Attach scores and sort
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            
        ranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_chunks[:top_k]
