from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env is at project root, two levels above this file (src/config.py → src/ → contextual_pipeline/ → project root)
_ENV_FILE = str(Path(__file__).parent.parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_batch_size: int = 32

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"

    context_version: int = 1
    llm_batch_size: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
