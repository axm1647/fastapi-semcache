# pyright: reportCallIssue=false

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class CacheSettings(BaseSettings):
    model_config = {"env_prefix": "SEMANTIC_CACHE_"}

    threshold: float = Field(
        0.90,
        description="Similarity threshold for cache hit (0.0–1.0)",
    )
    pg_uri: str = Field(
        "postgresql://user:pass@localhost:5433/semanticcache",
        description="PostgreSQL URI with pgvector extension",
    )
    redis_uri: str = Field(
        "redis://localhost:6379/0",
        description="Redis URI for cache TTL",
    )

    pg_pool_size: int = 10
    pg_pool_max_overflow: int = 20

    embedder_type: Literal["local", "openai"] = "local"

    class Config:
        env_file = ".env"  # optional env for openai API usage


@lru_cache(maxsize=1)
def get_cache_settings() -> CacheSettings:
    return CacheSettings()
