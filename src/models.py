from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

class GuardrailStatus(str, Enum):
    PASS = "pass"
    FAIL_SAFETY = "fail_safety"
    FAIL_OFFTOPIC = "fail_offtopic"
    FAIL_GROUNDING = "fail_grounding"
    FAIL_REFUSAL = "fail_refusal"

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    strategy: str
    language: str
    passage_id: str

class PipelineInput(BaseModel):
    query: str
    top_k: int = 5
    language: str = "en"

class PipelineOutput(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
    guardrail: GuardrailStatus
    guardrail_reason: Optional[str] = None
    latency: Dict[str, float]
