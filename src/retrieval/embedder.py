import os
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_id: str = "intfloat/multilingual-e5-base", onnx_path: str = "models/embedder.onnx"):
        self.model_id = model_id
        if os.path.exists(onnx_path):
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            from transformers import pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
            self.model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
            self.pipe = pipeline("feature-extraction", model=self.model, tokenizer=self.tokenizer)
            self.is_onnx = True
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_id)
            self.is_onnx = False
            
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = np.array(model_output[0])
        input_mask_expanded = np.expand_dims(np.array(attention_mask), -1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        prefixed = [f"query: {q}" for q in queries]
        return self._embed(prefixed)
        
    def embed_passages(self, passages: List[str]) -> np.ndarray:
        prefixed = [f"passage: {p}" for p in passages]
        return self._embed(prefixed)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries([query])[0]
        
    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.is_onnx:
            encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
            output = self.model(**encoded)
            embeddings = self._mean_pooling([output.last_hidden_state], encoded['attention_mask'])
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms
        else:
            return self.model.encode(texts, normalize_embeddings=True)
