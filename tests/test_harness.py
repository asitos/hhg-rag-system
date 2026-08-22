import pytest
import asyncio
from src.pipeline.harness import RAGPipeline
from src.models import PipelineInput, GuardrailStatus
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder
from src.config import settings

# Ensure we test in mock mode
settings.app_mode = "mock"
settings.demo_scenario = "english"

@pytest.fixture(scope="module")
def store():
    # Use real embedder for tests to verify actual pipeline
    embedder = Embedder(settings.embedding_model_id)
    vec = embedder.embed_query("A corporation is a legal entity.")
    
    vs = VectorStore()
    vs.add_vectors([vec], [{"chunk_id": "1", "text": "A corporation is a legal entity.", "strategy": "fixed", "language": "en", "passage_id": "p1"}])
    return vs

@pytest.fixture(scope="module")
def pipeline(store):
    p = RAGPipeline(store)
    return p

@pytest.mark.asyncio
async def test_harness_offtopic(pipeline):
    # Should fail pre-guardrail
    res = await pipeline.execute(PipelineInput(query="how to bake a cake"))
    assert res.guardrail == GuardrailStatus.FAIL_OFFTOPIC
    
@pytest.mark.asyncio
async def test_harness_no_context(pipeline):
    # Temporarily swap the store to avoid loading VRAM twice
    original_store = pipeline.vector_store
    pipeline.vector_store = VectorStore()
    
    res = await pipeline.execute(PipelineInput(query="What is a corporation?"))
    assert res.guardrail == GuardrailStatus.FAIL_GROUNDING
    
    # Restore
    pipeline.vector_store = original_store
    
@pytest.mark.asyncio
async def test_harness_success(pipeline):
    res = await pipeline.execute(PipelineInput(query="What is a corporation?"))
    assert res.guardrail == GuardrailStatus.PASS
    assert len(res.sources) >= 1

@pytest.mark.asyncio
async def test_harness_run_audio(pipeline):
    # Mock STT will return a deterministic string
    res = await pipeline.run(b"fake_audio_bytes")
    # Depends on which deterministic query MockSTT returns. 
    # "What is a corporation?" is in the rotation.
    # It might pass or fail grounding, but it should not crash.
    assert res.guardrail in [GuardrailStatus.PASS, GuardrailStatus.FAIL_GROUNDING]

@pytest.mark.asyncio
async def test_mock_failures(pipeline):
    # Test generation failure mode
    settings.demo_scenario = "grounding_failure"
    res = await pipeline.execute(PipelineInput(query="What is a corporation?"))
    # The post guardrail should catch the invalid citation
    assert res.guardrail == GuardrailStatus.FAIL_GROUNDING
    
    settings.demo_scenario = "english"
