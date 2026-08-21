import re
from typing import List
from .base import Chunker, Chunk

class ParagraphChunker(Chunker):
    """
    Splits text by double newlines or standard paragraph breaks.
    If a paragraph is excessively long, it forces a split.
    """
    def __init__(self, max_chars: int = 1000):
        self.max_chars = max_chars

    def chunk(self, text: str, passage_id: str, language: str) -> List[Chunk]:
        if not text.strip():
            return []
            
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks = []
        for i, para in enumerate(paragraphs):
            # Fallback if a paragraph is gigantic
            if len(para) > self.max_chars:
                for j in range(0, len(para), self.max_chars):
                    sub_para = para[j:j+self.max_chars]
                    chunks.append(Chunk(
                        chunk_id=f"{passage_id}_para_{i}_{j}",
                        text=sub_para,
                        strategy="paragraph",
                        metadata={"passage_id": passage_id, "language": language}
                    ))
            else:
                chunks.append(Chunk(
                    chunk_id=f"{passage_id}_para_{i}",
                    text=para,
                    strategy="paragraph",
                    metadata={"passage_id": passage_id, "language": language}
                ))
                
        return chunks
