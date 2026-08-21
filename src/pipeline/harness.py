import time
import logging
from typing import Dict, Any, List
from pydantic import BaseModel

from src.config import settings
from src.retrieval.embed import Embedder
from src.retrieval.index import VectorStore
from src.retrieval.rerank import Reranker
from src.guardrails.pre import PreGuardrail, GuardrailStatus
from src.guardrails.post import PostGuardrail
from src.generation.llm import Generator

logger = logging.getLogger(__name__)

class PipelineResponse(BaseModel):
    query: str
    answer: str
    status: str
    stage_latencies_ms: Dict[str, float]
    total_latency_ms: float
    retrieved_chunks: int

class RAGPipeline:
    """
    Explicit orchestration layer with typed/validated state and latency tracking.
    """
    def __init__(self, vector_store: VectorStore):
        self.embedder = Embedder(settings.embedding_model_id)
        self.vector_store = vector_store
        self.reranker = Reranker(settings.cross_encoder_id)
        self.pre_guard = PreGuardrail()
        self.post_guard = PostGuardrail()
        self.generator = Generator(settings.gemini_api_key)
        
    def execute(self, query: str) -> PipelineResponse:
        t_start = time.perf_counter()
        latencies = {}
        
        # 1. Pre-Retrieval Guardrail
        t0 = time.perf_counter()
        pre_status = self.pre_guard.check_query(query)
        latencies["guardrail_pre"] = (time.perf_counter() - t0) * 1000
        
        if pre_status != GuardrailStatus.PASS:
            return self._build_response(query, f"Blocked: {pre_status.value}", pre_status.value, latencies, t_start, 0)
            
        # 2. Embedding
        t0 = time.perf_counter()
        query_emb = self.embedder.embed_query(query)
        latencies["embedding"] = (time.perf_counter() - t0) * 1000
        
        # 3. Vector Retrieval (FAISS)
        t0 = time.perf_counter()
        # Retrieve slightly more for reranking
        candidate_chunks = self.vector_store.search(query_emb, top_k=15)
        latencies["faiss_search"] = (time.perf_counter() - t0) * 1000
        
        # 4. Reranking
        t0 = time.perf_counter()
        # BYPASS MONOLINGUAL RERANKER: ms-marco-MiniLM is English-only and penalizes Indic matches.
        # We rely purely on the strong multilingual alignments of e5-small.
        top_chunks = candidate_chunks[:5]
        latencies["reranking"] = (time.perf_counter() - t0) * 1000
        
        if not top_chunks:
            return self._build_response(query, "I don't have enough information in the retrieved context to answer that.", GuardrailStatus.FAIL_REFUSAL.value, latencies, t_start, 0)
            
        # 5. Generation
        t0 = time.perf_counter()
        answer, llm_latency = self.generator.generate(query, top_chunks)
        latencies["generation_llm"] = llm_latency
        
        # 6. Post-Generation Guardrail
        t0 = time.perf_counter()
        post_status = self.post_guard.check_grounding(answer, top_chunks)
        latencies["guardrail_post"] = (time.perf_counter() - t0) * 1000
        
        if post_status != GuardrailStatus.PASS:
            # Mask hallucination or pass refusal cleanly
            answer = "I don't have enough information in the retrieved context to answer that."
            status_val = post_status.value
        else:
            status_val = GuardrailStatus.PASS.value
            
        return self._build_response(query, answer, status_val, latencies, t_start, len(top_chunks))
        
    def _build_response(self, query: str, answer: str, status: str, latencies: Dict[str, float], t_start: float, chunks_count: int) -> PipelineResponse:
        total_latency = (time.perf_counter() - t_start) * 1000
        return PipelineResponse(
            query=query,
            answer=answer,
            status=status,
            stage_latencies_ms=latencies,
            total_latency_ms=total_latency,
            retrieved_chunks=chunks_count
        )
