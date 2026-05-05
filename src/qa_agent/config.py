from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Temporal
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    qa_task_queue: str = "qa-agent-tq"

    # Ollama embeddings
    ollama_base_url: str
    embedding_model: str

    # Cohere
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"

    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Retrieval knobs
    bm25_top_k: int = Field(default=50, ge=1, le=500)
    vector_top_k: int = Field(default=50, ge=1, le=500)
    rrf_top_n: int = Field(default=40, ge=1, le=200)
    max_expansion_total: int = Field(default=20, ge=0, le=100)
    rerank_top_k: int = Field(default=8, ge=1, le=50)
    rerank_doc_chars: int = Field(default=1500, ge=200, le=8000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
