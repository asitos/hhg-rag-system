import asyncio
from typing import List, Dict, Any, Tuple
from src.config import settings
import time

class MockGenerator:
    def __init__(self, api_key: str = None):
        pass

    async def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
        if settings.mock_failure_mode == "generation":
            raise Exception("Simulated generation failure")
            
        t0 = time.perf_counter()
        
        if settings.mock_latency:
            await asyncio.sleep(settings.mock_generation_latency_ms / 1000.0)
            
        if not context_chunks:
            ans = "I don't have enough information in the retrieved context to answer that."
            return ans, (time.perf_counter() - t0) * 1000
            
        if settings.mock_failure_mode == "ungrounded":
            ans = "This is an ungrounded answer with no citations at all. I am inventing facts."
            return ans, (time.perf_counter() - t0) * 1000
            
        if settings.mock_failure_mode == "invalid_citation":
            ans = "This is a deliberately invalid citation. [fake_chunk_id]"
            return ans, (time.perf_counter() - t0) * 1000
            
        if settings.mock_failure_mode == "no_context":
            ans = "I don't have enough information in the retrieved context to answer that."
            return ans, (time.perf_counter() - t0) * 1000

        # Normal successful grounded answer
        first_chunk = context_chunks[0]
        chunk_id = first_chunk.get("chunk_id", "unknown")
        text = first_chunk.get("text", "...")[:100]
        
        ans = f"Based on the retrieved context:\n\n[{chunk_id}]\n\"{text}...\"\n\nThis answer is generated in mock mode and is grounded only in the retrieved sources."
        return ans, (time.perf_counter() - t0) * 1000
