from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import uuid

class VectorStore:
    def __init__(self, dimension: int = 768):
        """
        Initializes an in-memory Qdrant index.
        Dimension 768 is standard for e5-base.
        """
        self.client = QdrantClient(":memory:")
        self.collection_name = "msmarco"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )

    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Adds normalized vectors to Qdrant with associated metadata."""
        if len(vectors) != len(metadatas):
            raise ValueError("Mismatched vectors and metadata lengths")
            
        points = []
        for vec, meta in zip(vectors, metadatas):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=meta
                )
            )
            
        # Batch upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i+batch_size]
            )

    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        strategy: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the index for the closest vectors, optionally filtering by strategy and language.
        """
        must_conditions = []
        if strategy:
            must_conditions.append(FieldCondition(key="strategy", match=MatchValue(value=strategy)))
        if language:
            must_conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
            
        query_filter = Filter(must=must_conditions) if must_conditions else None

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        ).points
        
        results = []
        for hit in search_result:
            meta = hit.payload.copy()
            meta["score"] = float(hit.score)
            results.append(meta)
            
        return results
