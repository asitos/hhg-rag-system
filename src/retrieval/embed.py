from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_id: str = "intfloat/multilingual-e5-small"):
        """
        Loads the SentenceTransformer model.
        multilingual-e5 requires 'query: ' and 'passage: ' prefixes for optimal performance.
        """
        self.model = SentenceTransformer(model_id)
        
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        # e5 requires 'query: ' prefix
        prefixed = [f"query: {q}" for q in queries]
        return self.model.encode(prefixed, normalize_embeddings=True)
        
    def embed_passages(self, passages: List[str]) -> np.ndarray:
        # e5 requires 'passage: ' prefix
        prefixed = [f"passage: {p}" for p in passages]
        return self.model.encode(prefixed, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries([query])[0]
