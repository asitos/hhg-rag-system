import re
import uuid
from typing import List
from .base import Chunker, Chunk

class SentenceChunker(Chunker):
    """
    Chunks text by sentence boundaries, attempting to group sentences up to a certain char limit.
    """
    def __init__(self, max_chars_per_chunk: int = 1000):
        self.max_chars_per_chunk = max_chars_per_chunk
        # Simple punctuation split for basic sentence detection
        self.split_pattern = re.compile(r'(?<=[.!?।])\s+')

    def chunk(self, text: str, passage_id: str, language: str) -> List[Chunk]:
        sentences = self.split_pattern.split(text)
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_len + len(sentence) > self.max_chars_per_chunk and current_chunk:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=" ".join(current_chunk),
                    strategy="sentence",
                    metadata={"passage_id": passage_id, "language": language}
                ))
                current_chunk = [sentence]
                current_len = len(sentence)
            else:
                current_chunk.append(sentence)
                current_len += len(sentence)
                
        if current_chunk:
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=" ".join(current_chunk),
                strategy="sentence",
                metadata={"passage_id": passage_id, "language": language}
            ))
            
        return chunks
