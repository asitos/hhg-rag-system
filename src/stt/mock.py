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
        if settings.demo_scenario == "off_topic":
            return "What is the weather today?"
        elif settings.demo_scenario == "no_context":
            return "Tell me about quantum physics."
        elif settings.demo_scenario == "hindi":
            return "भारत की राजधानी क्या है?"
        else:
            return "What is a corporation?"

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
