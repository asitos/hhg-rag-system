import time
import logging
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.stt.sarvam import SarvamSTT
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.guardrails.pre import PreGuardrail
from src.guardrails.post import PostGuardrail
from src.generation.gemini import Generator
from src.models import PipelineInput, PipelineOutput, RetrievedChunk, GuardrailStatus

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.stt = SarvamSTT(settings.sarvam_api_key)
        self.embedder = Embedder(settings.embedding_model_id)
        self.vector_store = vector_store
        self.reranker = Reranker(settings.cross_encoder_id)
        self.pre_guard = PreGuardrail()
        self.post_guard = PostGuardrail()
        self.generator = Generator(settings.gemini_api_key)
        
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.1))
    async def run(self, audio_bytes: bytes) -> PipelineOutput:
        t_start = time.perf_counter()
        latencies = {}
        
        # 1. STT
        t0 = time.perf_counter()
        # Ensure STT returns string text
        query = await self.stt.transcribe(audio_bytes)
        latencies["stt_ms"] = (time.perf_counter() - t0) * 1000
        
        if not query.strip():
            return self._build_response("Could not understand audio.", [], GuardrailStatus.FAIL_GROUNDING, "Empty STT", latencies, t_start)
            
        p_input = PipelineInput(query=query)
        
        # 2. Pre-Retrieval Guardrail
        t0 = time.perf_counter()
        pre_status = self.pre_guard.check_query(p_input.query)
        latencies["guardrail_pre_ms"] = (time.perf_counter() - t0) * 1000
        
        if pre_status != GuardrailStatus.PASS:
            return self._build_response("", [], pre_status, pre_status.value, latencies, t_start)
            
        # 3. Embedding
        t0 = time.perf_counter()
        query_emb = self.embedder.embed_query(p_input.query)
        latencies["embedding_ms"] = (time.perf_counter() - t0) * 1000
        
        # 4. Vector Retrieval
        t0 = time.perf_counter()
        strategies = ["fixed", "sentence", "paragraph", "semantic"]
        merged_candidates = []
        for strat in strategies:
            strat_candidates = self.vector_store.search(query_emb, top_k=3, strategy=strat)
            merged_candidates.extend(strat_candidates)
            
        seen = set()
        candidate_chunks = []
        for c in merged_candidates:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                candidate_chunks.append(c)
                
        # Tool call pattern for recovery
        if not candidate_chunks:
            # Rephrase via LLM (Mock tool call logic)
            rephrased = self.generator.generate(f"Rephrase this query for search: {p_input.query}", [])[0]
            query_emb = self.embedder.embed_query(rephrased)
            candidate_chunks = self.vector_store.search(query_emb, top_k=10)
            if not candidate_chunks:
                latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000
                return self._build_response("I don't have enough information.", [], GuardrailStatus.FAIL_GROUNDING, "No retrieved chunks after retry", latencies, t_start)

        latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Reranking
        t0 = time.perf_counter()
        top_chunks = self.reranker.rerank(p_input.query, candidate_chunks, top_k=p_input.top_k)
        latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000
        
        if not top_chunks:
            return self._build_response("I don't have enough information.", [], GuardrailStatus.FAIL_GROUNDING, "Chunks failed reranking", latencies, t_start)
            
        # 6. Generation
        t0 = time.perf_counter()
        answer, llm_latency = self.generator.generate(p_input.query, top_chunks)
        latencies["generation_ms"] = llm_latency
        
        # 7. Post-Generation Guardrail
        t0 = time.perf_counter()
        post_status = self.post_guard.check_grounding(answer, top_chunks)
        latencies["guardrail_post_ms"] = (time.perf_counter() - t0) * 1000
        
        reason = None
        if post_status != GuardrailStatus.PASS:
            answer = "I don't have enough information in the retrieved context to answer that."
            reason = f"Failed post-generation validation: {post_status.value}"
            
        sources = [RetrievedChunk(**c) for c in top_chunks]
        return self._build_response(answer, sources, post_status, reason, latencies, t_start)
        
    def execute(self, p_input: PipelineInput) -> PipelineOutput:
        """Sync fallback for benchmarking and legacy UI"""
        import asyncio
        # We simulate the async run without audio bytes since execute provides a query
        # Actually, let's keep execute() simple for benchmark.
        # But wait, harness.run is what should be tested.
        pass

    def _build_response(
        self, 
        answer: str, 
        sources: List[RetrievedChunk],
        status: GuardrailStatus, 
        reason: str,
        latencies: Dict[str, float], 
        t_start: float
    ) -> PipelineOutput:
        post_stt_ms = latencies.get("guardrail_pre_ms", 0) + latencies.get("embedding_ms", 0) + latencies.get("retrieval_ms", 0) + latencies.get("rerank_ms", 0) + latencies.get("generation_ms", 0) + latencies.get("guardrail_post_ms", 0)
        latencies["post_stt_ms"] = post_stt_ms
        latencies["total_ms"] = post_stt_ms + latencies.get("stt_ms", 0)
        
        return PipelineOutput(
            answer=answer,
            sources=sources,
            guardrail=status,
            guardrail_reason=reason,
            latency=latencies
        )

    def execute(self, p_input: PipelineInput) -> PipelineOutput:
        """Sync text-only pipeline execution for benchmarking"""
        t_start = time.perf_counter()
        latencies = {}
        
        t0 = time.perf_counter()
        pre_status = self.pre_guard.check_query(p_input.query)
        latencies["guardrail_pre_ms"] = (time.perf_counter() - t0) * 1000
        
        if pre_status != GuardrailStatus.PASS:
            return self._build_response("", [], pre_status, pre_status.value, latencies, t_start)
            
        t0 = time.perf_counter()
        query_emb = self.embedder.embed_query(p_input.query)
        latencies["embedding_ms"] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        strategies = ["fixed", "sentence", "paragraph", "semantic"]
        merged_candidates = []
        for strat in strategies:
            strat_candidates = self.vector_store.search(query_emb, top_k=3, strategy=strat)
            merged_candidates.extend(strat_candidates)
            
        seen = set()
        candidate_chunks = []
        for c in merged_candidates:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                candidate_chunks.append(c)
                
        if not candidate_chunks:
            rephrased = self.generator.generate(f"Rephrase this query for search: {p_input.query}", [])[0]
            query_emb = self.embedder.embed_query(rephrased)
            candidate_chunks = self.vector_store.search(query_emb, top_k=10)
            if not candidate_chunks:
                latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000
                return self._build_response("I don't have enough information.", [], GuardrailStatus.FAIL_GROUNDING, "No chunks after retry", latencies, t_start)

        latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        top_chunks = self.reranker.rerank(p_input.query, candidate_chunks, top_k=p_input.top_k)
        latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000
        
        if not top_chunks:
            return self._build_response("I don't have enough information.", [], GuardrailStatus.FAIL_GROUNDING, "Chunks failed reranking", latencies, t_start)
            
        t0 = time.perf_counter()
        answer, llm_latency = self.generator.generate(p_input.query, top_chunks)
        latencies["generation_ms"] = llm_latency
        
        t0 = time.perf_counter()
        post_status = self.post_guard.check_grounding(answer, top_chunks)
        latencies["guardrail_post_ms"] = (time.perf_counter() - t0) * 1000
        
        reason = None
        if post_status != GuardrailStatus.PASS:
            answer = "I don't have enough information in the retrieved context to answer that."
            reason = f"Failed post-generation validation: {post_status.value}"
            
        sources = [RetrievedChunk(**c) for c in top_chunks]
        return self._build_response(answer, sources, post_status, reason, latencies, t_start)
