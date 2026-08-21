from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Dict, Any

class Chunk(BaseModel):
    chunk_id: str
    text: str
    strategy: str
    metadata: Dict[str, Any]

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, passage_id: str, language: str) -> List[Chunk]:
        """Convert a passage text into a list of Chunk objects."""
        pass
