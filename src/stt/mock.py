import asyncio
from src.config import settings

class MockSTT:
    def __init__(self):
        self.queries = [
            "What is a corporation?",
            "What is the capital of India?",
            "How do I invest in stocks?",
            "Tell me about mutual funds."
        ]
        self.index = 0

    async def transcribe(self, audio_bytes: bytes, language="unknown") -> str:
        if settings.mock_failure_mode == "stt":
            raise Exception("Simulated STT failure")
            
        if settings.mock_failure_mode == "timeout":
            await asyncio.sleep(10)
            raise TimeoutError("Simulated timeout")
            
        if settings.mock_latency:
            await asyncio.sleep(settings.mock_stt_latency_ms / 1000.0)
            
        # Very simple deterministic rotation
        query = self.queries[self.index % len(self.queries)]
        self.index += 1
        return query

    async def close(self):
        pass

class MockStreamSession:
    def __init__(self):
        self.latest_partial = ""
        self.is_final = False
        
    async def connect(self):
        return True
        
    async def send_chunk(self, chunk):
        pass
        
    async def finalize(self):
        if settings.mock_failure_mode == "stt":
            raise Exception("Simulated STT failure")
        return "Mock streamed query", {"stt_final_ms": 100}
        
    async def close(self):
        pass
