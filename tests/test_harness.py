import pytest
from src.pipeline.harness import RAGPipeline
from src.models import PipelineInput, GuardrailStatus
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder
import numpy as np

# We mock external APIs to avoid needing real keys
class MockEmbedder:
    def embed_query(self, query):
        return np.random.rand(768).tolist()

class MockReranker:
    def rerank(self, query, chunks, top_k):
        return chunks[:top_k]

class MockGenerator:
    def generate(self, query, chunks):
        return "This is a mock answer. [1]", 100.0

def test_harness_offtopic():
    # Setup mock
    store = VectorStore()
    pipeline = RAGPipeline(store)
    
    # Should fail pre-guardrail
    res = pipeline.execute(PipelineInput(query="how to bake a cake"))
    assert res.guardrail == GuardrailStatus.FAIL_OFFTOPIC
    
def test_harness_no_context():
    store = VectorStore()
    pipeline = RAGPipeline(store)
    pipeline.embedder = MockEmbedder()
    pipeline.reranker = MockReranker()
    pipeline.generator = MockGenerator()
    
    # Empty store should fail grounding recovery
    res = pipeline.execute(PipelineInput(query="What is a corporation?"))
    assert res.guardrail == GuardrailStatus.FAIL_GROUNDING
    
def test_harness_success():
    store = VectorStore()
    store.add_vectors([np.random.rand(768).tolist()], [{"chunk_id": "1", "text": "corporation", "strategy": "fixed", "language": "en", "passage_id": "p1"}])
    
    pipeline = RAGPipeline(store)
    pipeline.embedder = MockEmbedder()
    pipeline.reranker = MockReranker()
    pipeline.generator = MockGenerator()
    
    res = pipeline.execute(PipelineInput(query="What is a corporation?"))
    assert res.guardrail == GuardrailStatus.PASS
    assert len(res.sources) == 1
