from enum import Enum
import re

class GuardrailStatus(str, Enum):
    PASS = "pass"
    FAIL_SAFETY = "fail_safety"
    FAIL_OFFTOPIC = "fail_offtopic"
    FAIL_GROUNDING = "fail_grounding"
    FAIL_REFUSAL = "fail_refusal"

class PreGuardrail:
    """
    Executes before retrieval to block unsafe or completely out-of-domain queries.
    """
    def __init__(self):
        # A fast regex filter for obvious jailbreaks or harmful content
        self.unsafe_patterns = re.compile(
            r'(ignore all previous instructions|jailbreak|system prompt|hack|bypass)', 
            re.IGNORECASE
        )
        
    def check_query(self, query: str) -> GuardrailStatus:
        if not query or len(query.strip()) < 2:
            return GuardrailStatus.FAIL_OFFTOPIC
            
        if self.unsafe_patterns.search(query):
            return GuardrailStatus.FAIL_SAFETY
            
        return GuardrailStatus.PASS
