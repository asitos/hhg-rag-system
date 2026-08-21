import pytest
from src.retrieval.vector_store import VectorStore
import numpy as np

def test_vector_store():
    store = VectorStore(dimension=10)
    
    vec1 = np.random.rand(10).tolist()
    vec2 = np.random.rand(10).tolist()
    
    store.add_vectors([vec1, vec2], [{"chunk_id": "1", "strategy": "fixed", "language": "en"}, 
                                    {"chunk_id": "2", "strategy": "sentence", "language": "hi"}])
                                    
    # Search without filter
    res = store.search(vec1, top_k=2)
    assert len(res) == 2
    
    # Search with filter
    res = store.search(vec1, top_k=2, strategy="sentence", language="hi")
    assert len(res) == 1
    assert res[0]["chunk_id"] == "2"
