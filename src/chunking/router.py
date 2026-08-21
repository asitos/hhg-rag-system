from typing import List
from .base import Chunker, Chunk
from .fixed import FixedSizeChunker
from .sentence import SentenceChunker
from .paragraph import ParagraphChunker
from .semantic import SemanticChunker

class ChunkingRouter:
    """
    Applies multiple chunking strategies simultaneously to preserve different structural meanings.
    """
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunkers: List[Chunker] = [
            FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            SentenceChunker(max_chars_per_chunk=chunk_size * 4),
            ParagraphChunker(),
            SemanticChunker()
        ]
        
    def process_passage(self, text: str, passage_id: str, language: str) -> List[Chunk]:
        """Runs all configured chunkers and aggregates the chunks."""
        all_chunks = []
        for chunker in self.chunkers:
            all_chunks.extend(chunker.chunk(text, passage_id, language))
        return all_chunks
