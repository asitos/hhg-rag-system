import time
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential

class Generator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Using 3.6-flash because 2.0-flash was deprecated and sunset by the API.
        self.model_id = "gemini-3.6-flash"
        
        self.system_prompt = (
            "You are a strict retrieval-augmented generation assistant.\n"
            "Answer ONLY from the supplied context.\n"
            "Do not invent information.\n"
            "If the context does not support the answer, explicitly refuse by returning EXACTLY: 'I don't have enough information in the retrieved context to answer that.'\n"
            "Cite every factual claim using [chunk_id]."
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=4))
    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Generates an answer based on retrieved context.
        Returns: (answer_text, api_latency_ms)
        """
        context_str = "\n\n".join(
            f"[{c['passage_id']}] {c['text']}" for c in context_chunks
        )
        
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        t0 = time.perf_counter()
        chat = self.client.chats.create(model=self.model_id)
        # In a real deployed UI, we would use send_message_stream to get TTFT < 200ms.
        # For this harness, we await the full generation and track the absolute API latency.
        response = chat.send_message([self.system_prompt, prompt])
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return response.text.strip(), latency_ms
