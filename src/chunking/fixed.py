from typing import List
from .base import Chunker, Chunk
import uuid

class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, passage_id: str, language: str) -> List[Chunk]:
        chunks = []
        # Fallback to simple word/character chunking since proper tokenization is heavy
        # Assuming avg 5 chars per token for English/Indic mix
        char_size = self.chunk_size * 5
        char_overlap = self.chunk_overlap * 5
        
        start = 0
        text_len = len(text)
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(Chunk(
                chunk_id=f"{passage_id}_fixed_{i}",
                text=chunk_text,
                strategy="fixed",
                metadata={"passage_id": passage_id, "language": language}
            ))
            
        return chunks
