import time
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential

class Generator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3.5-flash"
        
        self.system_prompt = (
            "You are a strict retrieval-augmented generation assistant.\n"
            "You must answer the user's question using ONLY the provided context.\n"
            "If the context does not contain the answer, you must output EXACTLY: 'I don't have enough information in the retrieved context to answer that.'\n"
            "If you use information from the context, cite the chunk ID at the end of the sentence like [p_123]."
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
