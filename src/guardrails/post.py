import re
from typing import List, Dict, Any
from .pre import GuardrailStatus

class PostGuardrail:
    """
    Executes after LLM generation to catch hallucinations and verify grounding.
    """
    def check_grounding(self, answer: str, context_chunks: List[Dict[str, Any]]) -> GuardrailStatus:
        answer_lower = answer.lower()
        
        # 1. Check for intentional refusal from the LLM
        refusal_phrases = [
            "i don't have enough information",
            "i cannot answer",
            "the context does not provide",
            "no relevant information",
            "insufficient information"
        ]
        for phrase in refusal_phrases:
            if phrase in answer_lower:
                return GuardrailStatus.FAIL_REFUSAL
                
        # 2. Heuristic Grounding Check (Citation tracking)
        # If the LLM was instructed to cite chunks like [p_123], ensure they exist in context
        cited_ids = set(re.findall(r'\[(p_\d+)\]', answer))
        context_ids = {c["passage_id"] for c in context_chunks}
        
        # If citations were found but none exist in the context, it's a hallucination
        if cited_ids and not cited_ids.intersection(context_ids):
            return GuardrailStatus.FAIL_GROUNDING
            
        # In a full production system, we'd run an NLI (Natural Language Inference) model here
        # or do a strong lexical overlap check. For latency (<200ms), we rely on strict LLM prompting
        # combined with citation checks.
        
        return GuardrailStatus.PASS
