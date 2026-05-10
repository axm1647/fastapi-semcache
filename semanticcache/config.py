# pyright: reportCallIssue=false

"""Environment-backed cache settings (Postgres, Redis, embedder selection)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import AliasChoices, Field, model_validator
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
            "matches after initially fetching top_k_candidates. Must be >= "
            "threshold when set, or validation fails. When unset, threshold alone "
            "controls cache hits."
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
            "When set, also configures Redis socket_timeout and "
            "socket_connect_timeout for the response store client. "
            "Set to null/empty to disable asyncio timeouts and Redis socket timeouts."
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
    voyage_embedding_model: str | None = Field(
        default=None,
        description=(
            "Voyage embedding model id; used when embedder_type is voyage. "
            "Defaults to 'voyage-3' when unset."
        ),
    )
    voyage_embedding_dimensions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Embedding vector width for the configured Voyage model; used when "
            "embedder_type is voyage. Defaults to 1024 when unset."
        ),
    )
    voyage_input_type: str | None = Field(
        default=None,
        description=(
            "Optional Voyage input_type hint sent with each embed request. "
            "Options: None, 'query', 'document'. Used when embedder_type is voyage."
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
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434/v1",
        description=(
            "OpenAI-compatible API root for Ollama (path must include /v1). "
            "Used when embedder_type is ollama."
        ),
    )
    ollama_embedding_model: str | None = Field(
        default=None,
        description=(
            "Ollama embedding model id; required when embedder_type is ollama "
            "(must match ``ollama_embedding_dimensions`` and your pgvector column)."
        ),
    )
    ollama_embedding_dimensions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Embedding vector width for the configured Ollama model; required when "
            "embedder_type is ollama."
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
    middleware_flight_lock_max_entries: int = Field(
        4096,
        ge=1,
        description=(
            "Maximum number of distinct in-flight middleware lock keys retained for "
            "coordinating concurrent misses. Oldest unlocked entries are evicted."
        ),
    )
    require_cache_scope: bool = Field(
        True,
        description=(
            "When True, ``SemanticCache`` and middleware require a non-empty tenant "
            "or namespace scope for lookups and writes (see ``resolve_cache_scope``). "
            "Set False only for single-tenant deployments or dedicated cache storage."
        ),
    )
    cache_authorized_requests: bool = Field(
        False,
        description=(
            "When True, middleware may cache requests that include an Authorization "
            "header. Default False to avoid accidental cross-user response reuse."
        ),
    )

    @model_validator(mode="after")
    def _validate_ollama_embedding_settings(self) -> CacheSettings:
        """Ensure Ollama embedder settings include model id and vector width.

        Returns:
            Unchanged settings when validation passes.

        Raises:
            ValueError: When ``embedder_type`` is ollama but model or dimensions are
                missing.
        """
        if self.embedder_type != "ollama":
            return self
        if (
            self.ollama_embedding_model is None
            or not self.ollama_embedding_model.strip()
        ):
            msg = (
                "ollama_embedding_model is required when embedder_type is 'ollama' "
                "(set SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL)."
            )
            raise ValueError(msg)
        if self.ollama_embedding_dimensions is None:
            msg = (
                "ollama_embedding_dimensions is required when embedder_type is "
                "'ollama' (set SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS)."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_rejection_threshold_vs_primary(self) -> CacheSettings:
        """Ensure the rejection gate can filter stage-1 candidates when enabled.

        Returns:
            Unchanged settings when validation passes.

        Raises:
            ValueError: When ``rejection_threshold`` is set but lower than
                ``threshold`` (second stage could not reject any stage-1 hit).
        """
        if self.rejection_threshold is None:
            return self
        if self.rejection_threshold < self.threshold:
            msg: str = (
                "rejection_threshold must be >= threshold when set "
                f"(got rejection_threshold={self.rejection_threshold!r}, "
                f"threshold={self.threshold!r}). "
                "Otherwise the second stage cannot reject any candidate that passed "
                "the primary similarity gate."
            )
            raise ValueError(msg)
        return self


def get_cache_settings() -> CacheSettings:
    """Load ``CacheSettings`` from process environment variables.

    Variables use the ``SEMANTIC_CACHE_`` prefix (see ``CacheSettings`` fields).

    Returns:
        Validated settings instance.
    """
    return CacheSettings()
