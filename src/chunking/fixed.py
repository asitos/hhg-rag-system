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
        
        while start < text_len:
            end = min(start + char_size, text_len)
            chunk_text = text[start:end]
            
            # Avoid tiny trailing chunks
            if len(chunk_text) < 20 and chunks:
                break
                
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                strategy="fixed",
                metadata={"passage_id": passage_id, "language": language}
            ))
            
            if end == text_len:
                break
                
            start += char_size - char_overlap
            
        return chunks
