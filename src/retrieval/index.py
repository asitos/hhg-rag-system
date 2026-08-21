import faiss
import numpy as np
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        Initializes an in-memory FAISS index (HNSW for speed).
        Dimension 384 is standard for e5-small.
        """
        # HNSW provides extreme search speeds at the cost of slightly slower build time
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efSearch = 64
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self._current_id = 0

    def add_vectors(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Adds normalized vectors to the FAISS index with associated metadata."""
        if len(vectors) != len(metadatas):
            raise ValueError("Mismatched vectors and metadata lengths")
            
        # FAISS requires float32
        vectors = np.array(vectors, dtype=np.float32)
        
        self.index.add(vectors)
        
        for meta in metadatas:
            self.metadata[self._current_id] = meta
            self._current_id += 1

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the index for the closest vectors (cosine similarity via inner product 
        since vectors are normalized, but HNSWFlat uses L2 by default. Since vectors 
        are normalized, L2 distance ordering is identical to inner product ordering).
        """
        # Ensure correct shape and type
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        query_vector = np.array(query_vector, dtype=np.float32)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx not in self.metadata:
                continue
            
            # Convert L2 distance to something resembling a similarity score
            # For L2 squared on normalized vectors: L2^2 = 2 - 2 * cosine_sim
            # cosine_sim = 1 - (L2^2 / 2)
            cosine_sim = 1.0 - (distances[0][i] / 2.0)
            
            meta = self.metadata[idx].copy()
            meta["score"] = float(cosine_sim)
            results.append(meta)
            
        return results
