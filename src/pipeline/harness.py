import time
import logging
from typing import Dict, Any, List

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.guardrails.pre import PreGuardrail
from src.guardrails.post import PostGuardrail
from src.generation.gemini import Generator
from src.models import PipelineInput, PipelineOutput, RetrievedChunk, GuardrailStatus

logger = logging.getLogger(__name__)

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
        
    def execute(self, p_input: PipelineInput) -> PipelineOutput:
        t_start = time.perf_counter()
        latencies = {}
        
        # 1. Pre-Retrieval Guardrail
        t0 = time.perf_counter()
        pre_status = self.pre_guard.check_query(p_input.query)
        latencies["guardrail_pre_ms"] = (time.perf_counter() - t0) * 1000
        
        if pre_status != GuardrailStatus.PASS:
            return self._build_response("", [], pre_status, pre_status.value, latencies, t_start)
            
        # 2. Embedding
        t0 = time.perf_counter()
        query_emb = self.embedder.embed_query(p_input.query)
        latencies["embedding_ms"] = (time.perf_counter() - t0) * 1000
        
        # 3. Vector Retrieval (Multi-strategy)
        t0 = time.perf_counter()
        # Retrieve across all strategies and merge
        strategies = ["fixed", "sentence", "paragraph", "semantic"]
        merged_candidates = []
        for strat in strategies:
            strat_candidates = self.vector_store.search(query_emb, top_k=3, strategy=strat)
            merged_candidates.extend(strat_candidates)
            
        # Deduplicate
        seen = set()
        candidate_chunks = []
        for c in merged_candidates:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                candidate_chunks.append(c)
                
        latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        
        # Recovery loop if no chunks
        if not candidate_chunks:
            # Rephrase or retry without strategy filters
            candidate_chunks = self.vector_store.search(query_emb, top_k=10)
            if not candidate_chunks:
                return self._build_response("I don't have enough information.", [], GuardrailStatus.FAIL_GROUNDING, "No retrieved chunks", latencies, t_start)

        # 4. Reranking
        t0 = time.perf_counter()
        # Note: ms-marco cross encoder is English primary. It might penalize Indic languages.
        top_chunks = self.reranker.rerank(p_input.query, candidate_chunks, top_k=p_input.top_k)
        latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000
        
        if not top_chunks:
            return self._build_response("I don't have enough information.", [], GuardrailStatus.FAIL_GROUNDING, "Chunks failed reranking", latencies, t_start)
            
        # 5. Generation
        t0 = time.perf_counter()
        answer, llm_latency = self.generator.generate(p_input.query, top_chunks)
        latencies["generation_ms"] = llm_latency
        
        # 6. Post-Generation Guardrail
        t0 = time.perf_counter()
        post_status = self.post_guard.check_grounding(answer, top_chunks)
        latencies["guardrail_post_ms"] = (time.perf_counter() - t0) * 1000
        
        reason = None
        if post_status != GuardrailStatus.PASS:
            answer = "I don't have enough information in the retrieved context to answer that."
            reason = f"Failed post-generation validation: {post_status.value}"
            
        # Format sources
        sources = [RetrievedChunk(**c) for c in top_chunks]
        
        return self._build_response(answer, sources, post_status, reason, latencies, t_start)
        
    def _build_response(
        self, 
        answer: str, 
        sources: List[RetrievedChunk],
        status: GuardrailStatus, 
        reason: str,
        latencies: Dict[str, float], 
        t_start: float
    ) -> PipelineOutput:
        # Calculate post-stt latency
        post_stt_ms = sum(latencies.values())
        latencies["post_stt_ms"] = post_stt_ms
        latencies["total_ms"] = post_stt_ms # Stt added later
        
        return PipelineOutput(
            answer=answer,
            sources=sources,
            guardrail=status,
            guardrail_reason=reason,
            latency=latencies
        )
