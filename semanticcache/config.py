# pyright: reportCallIssue=false

"""Environment-backed cache settings (Postgres, Redis, embedder selection)."""

from typing import ClassVar, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheSettings(BaseSettings):
    """Load cache configuration from process environment variables only."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="SEMANTIC_CACHE_",
        env_file=None,
        extra="ignore",
    )

    disable_proxy_app_docs: bool = True  # hides documentation urls for proxy app

    top_k_candidates: int = Field(
        1,
        description=(
            "Maximum number of nearest-neighbor candidates fetched before applying "
            "the final similarity decision step."
        ),
        ge=1,
    )

    threshold: float = Field(
        0.90,
        description="Primary similarity threshold for candidate inclusion (0.0-1.0).",
        ge=0.0,  # guard boundaries
        le=1.0,
    )
    rejection_threshold: float | None = Field(
        default=None,
        description=(
            "Optional second-stage similarity threshold used to reject borderline "
            "matches after initially fetching top_k_candidates. When unset, "
            "threshold alone controls cache hits."
        ),
        ge=0.0,
        le=1.0,
    )
    pg_uri: str = Field(
        "postgresql://user:pass@localhost:5432/semanticcache",
        description="PostgreSQL URI with pgvector extension",
    )
    redis_uri: str = Field(
        "",
        description=(
            "Redis URI for cache TTL (empty/whitespace disables Redis and uses "
            "Postgres-only response storage)"
        ),
    )
    redis_ttl_seconds: int = Field(
        3600,
        description="Default TTL for Redis-cached responses (seconds)",
    )

    pg_pool_size: int = 10
    pg_pool_max_overflow: int = 20
    embed_timeout_seconds: float | None = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Timeout budget for embedder calls in seconds. Set to null/empty to disable."
        ),
    )
    store_timeout_seconds: float | None = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Timeout budget for Postgres/Redis operations in seconds. "
            "Set to null/empty to disable."
        ),
    )

    embedder_type: Literal["huggingface", "openai", "cohere", "voyage", "ollama"] = (
        "huggingface"
    )
    hugging_face_api_key: str | None = Field(
        default=None,
        description="Hugging Face API key",
        validation_alias=AliasChoices(
            "HUGGINGFACE_API_KEY",
            "SEMANTIC_CACHE_HUGGING_FACE_API_KEY",
        ),
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "SEMANTIC_CACHE_OPENAI_API_KEY",
        ),
    )
    cohere_api_key: str | None = Field(
        default=None,
        description="Cohere API key",
        validation_alias=AliasChoices(
            "COHERE_API_KEY",
            "SEMANTIC_CACHE_COHERE_API_KEY",
        ),
    )
    voyage_api_key: str | None = Field(
        default=None,
        description="Voyage API key",
        validation_alias=AliasChoices(
            "VOYAGE_API_KEY",
            "SEMANTIC_CACHE_VOYAGE_API_KEY",
        ),
    )
    ollama_api_key: str | None = Field(
        default=None,
        description="Ollama API key",
        validation_alias=AliasChoices(
            "OLLAMA_API_KEY",
            "SEMANTIC_CACHE_OLLAMA_API_KEY",
        ),
    )

    circuit_breaker_429_enabled: bool = Field(
        False,
        description=(
            "When True, after enough consecutive upstream HTTP 429 responses the "
            "middleware stops forwarding and only serves cache hits until cooldown."
        ),
    )
    circuit_breaker_429_consecutive_limit: int = Field(
        5,
        ge=1,
        description="Number of consecutive 429 responses required to open the circuit.",
    )
    circuit_breaker_429_open_seconds: float = Field(
        60.0,
        gt=0,
        description=(
            "Seconds to keep the circuit open (cache-only); after this, upstream "
            "requests resume."
        ),
    )


def get_cache_settings() -> CacheSettings:
    """Load ``CacheSettings`` from process environment variables.

    Variables use the ``SEMANTIC_CACHE_`` prefix (see ``CacheSettings`` fields).

    Returns:
        Validated settings instance.
    """
    return CacheSettings()
