from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Sales Knowledge RAG Assistant"
    log_level: str = "INFO"
    log_json: bool = True

    # Vector store
    vector_backend: Literal["pgvector", "memory"] = "pgvector"
    database_url: str | None = None  # e.g. postgresql+psycopg://rag:rag@localhost:5432/rag
    collection_name: str = "sales_docs"

    # Models
    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    fallback_chat_model: str = "gpt-4.1-nano"
    embedding_provider: Literal["auto", "openai", "hashing"] = "auto"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # Ingestion / retrieval
    docs_dir: str = "data/docs"
    auto_ingest: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_relevance: float = 0.0
    source_base_url: str = "https://github.com/sanskar-ucal/Sales-Knowledge-RAG-Assistant/blob/main"

    # Agent
    agent_max_steps: int = 4

    # Caching
    cache_ttl_seconds: int = 600
    cache_max_items: int = 512

    @property
    def use_openai_embeddings(self) -> bool:
        if self.embedding_provider == "openai":
            return True
        if self.embedding_provider == "hashing":
            return False
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
