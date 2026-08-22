from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    app_mode: str = "mock"
    mock_failure_mode: str = "none"
    mock_latency: bool = False
    mock_stt_latency_ms: int = 100
    mock_generation_latency_ms: int = 300
    
    # API Keys
    sarvam_api_key: str = ""
    gemini_api_key: str = ""
    frontend_origin: str = ""

    # Supported Languages (ISO codes for the 14 Indic + English)
    supported_languages: List[str] = [
        "en", "hi", "ta", "te", "bn", "ur", "mr", "gu", 
        "kn", "ml", "pa", "as", "or", "ne", "sa"
    ]

    # Retrieval configs
    embedding_model_id: str = "intfloat/multilingual-e5-base"
    cross_encoder_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking configs
    chunk_size: int = 256
    chunk_overlap: int = 32

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
