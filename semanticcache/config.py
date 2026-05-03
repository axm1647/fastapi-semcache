# pyright: reportCallIssue=false

from functools import lru_cache
from typing import ClassVar, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="SEMANTIC_CACHE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    threshold: float = Field(
        0.90,
        description="Similarity threshold for cache hit (0.0–1.0)",
    )
    pg_uri: str = Field(
        "postgresql://user:pass@localhost:5432/semanticcache",
        description="PostgreSQL URI with pgvector extension",
    )
    redis_uri: str = Field(
        "redis://localhost:6379/0",
        description="Redis URI for cache TTL",
    )
    redis_ttl_seconds: int = Field(
        3600,
        description="Default TTL for Redis-cached responses (seconds)",
    )

    pg_pool_size: int = 10
    pg_pool_max_overflow: int = 20

    embedder_type: Literal["local", "openai"] = "local"


@lru_cache(maxsize=1)
def get_cache_settings() -> CacheSettings:
    return CacheSettings()
