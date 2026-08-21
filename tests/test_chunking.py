import pytest
from src.chunking.router import ChunkingRouter

def test_chunking_router():
    router = ChunkingRouter(chunk_size=100, chunk_overlap=20)
    text = "This is a sentence. This is another sentence. " * 20
    chunks = router.process_passage(text, "p_1", "en")
    
    assert len(chunks) > 0
    
    strategies = {c.strategy for c in chunks}
    assert "fixed" in strategies
    assert "sentence" in strategies
    assert "paragraph" in strategies
    assert "semantic" in strategies
    
    for c in chunks:
        assert c.chunk_id.startswith("p_1")
        assert c.metadata["passage_id"] == "p_1"
        assert c.metadata["language"] == "en"
