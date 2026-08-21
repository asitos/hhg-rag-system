import re
from src.models import GuardrailStatus
from sentence_transformers import SentenceTransformer
import numpy as np

class PreGuardrail:
    """
    Executes before retrieval to block unsafe or completely out-of-domain queries.
    """
    def __init__(self):
        self.unsafe_patterns = re.compile(
            r'(ignore all previous instructions|jailbreak|system prompt|hack|bypass)', 
            re.IGNORECASE
        )
        
        # Domain relevance heuristic
        self.offtopic_patterns = re.compile(
            r'(recipe|bake|cake|cookie|weather|movie|actor|sports|football)', 
            re.IGNORECASE
        )
        
    def check_query(self, query: str) -> GuardrailStatus:
        if not query or len(query.strip()) < 2:
            return GuardrailStatus.FAIL_OFFTOPIC
            
        if self.unsafe_patterns.search(query):
            return GuardrailStatus.FAIL_SAFETY
            
        if self.offtopic_patterns.search(query):
            return GuardrailStatus.FAIL_OFFTOPIC
            
        return GuardrailStatus.PASS
