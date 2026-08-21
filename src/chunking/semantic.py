import re
from typing import List
from .base import Chunker, Chunk
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import settings

class SemanticChunker(Chunker):
    """
    Groups sentences into chunks based on semantic similarity.
    Splits when a topic shift is detected (cosine similarity drops below threshold).
    Because ingestion is offline, we can afford the overhead of an embedding model here.
    """
    def __init__(self, threshold: float = 0.65, max_sentences_per_chunk: int = 5):
        self.threshold = threshold
        self.max_sentences = max_sentences_per_chunk
        # Using a very fast/small model strictly for calculating shift gradients during chunking
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk(self, text: str, passage_id: str, language: str) -> List[Chunk]:
        if not text.strip():
            return []
            
        # Basic sentence splitting
        sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', text) if len(s.strip()) > 5]
        if not sentences:
            sentences = [text]
            
        if len(sentences) == 1:
            return [Chunk(
                chunk_id=f"{passage_id}_sem_0",
                text=sentences[0],
                strategy="semantic",
                metadata={"passage_id": passage_id, "language": language}
            )]

        # Get embeddings for all sentences to compute adjacent similarities
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-9
            )
            
            # Topic shift detected OR reached max length
            if sim < self.threshold or len(current_chunk) >= self.max_sentences:
                chunks.append(Chunk(
                    chunk_id=f"{passage_id}_sem_{len(chunks)}",
                    text=" ".join(current_chunk),
                    strategy="semantic",
                    metadata={"passage_id": passage_id, "language": language}
                ))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
                
        if current_chunk:
            chunks.append(Chunk(
                chunk_id=f"{passage_id}_sem_{len(chunks)}",
                text=" ".join(current_chunk),
                strategy="semantic",
                metadata={"passage_id": passage_id, "language": language}
            ))
            
        return chunks
