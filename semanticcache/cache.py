"""High-level semantic cache orchestrating embedder, pgvector, and optional Redis."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import TYPE_CHECKING, Awaitable, Literal, TypeVar

from .config import get_cache_settings
from .embedders import BaseEmbedder, get_embedder
from .exceptions import CacheTimeoutError
from .stores import AsyncPgVectorStore, RedisResponseStore
from .stores.vector.storage_ids import embedding_storage_ids
from .types import CacheResult

if TYPE_CHECKING:
    from semanticcache.config import CacheSettings

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _embed_source(
    settings: "CacheSettings",
) -> Literal[
    "embedders.sbert",
    "embedders.openai",
    "embedders.cohere",
    "embedders.voyage",
    "embedders.ollama",
    "none",
]:
    """Map configured embedder type to ``CacheResult.source``."""
    if settings.embedder_type == "huggingface":
        return "embedders.sbert"
    if settings.embedder_type == "openai":
        return "embedders.openai"
    if settings.embedder_type == "cohere":
        return "embedders.cohere"
    if settings.embedder_type == "voyage":
        return "embedders.voyage"
    if settings.embedder_type == "ollama":
        return "embedders.ollama"
    return "none"


class SemanticCache:
    """Orchestrate embedding, pgvector ANN search, and optional Redis response cache."""

    threshold: float
    pg_uri: str
    redis_uri: str
    _embedding_dim: int
    _redis_key_prefix: str
    _settings: "CacheSettings"
    _embedder: BaseEmbedder
    _vector_store: AsyncPgVectorStore
    _pg_open: bool
    _closed: bool
    _embed_timeout_seconds: float | None
    _store_timeout_seconds: float | None
    _timeout_counts: Counter[str]

    def __init__(
        self,
        threshold: float | None = None,
        pg_uri: str | None = None,
        redis_uri: str | None = None,
        *,
        embedder: BaseEmbedder | None = None,
        embedding_dim: int | None = None,
        settings: "CacheSettings | None" = None,
    ) -> None:
        """Build stores and embedder from explicit args or ``CacheSettings``.

        Args:
            threshold: Minimum cosine similarity in ``[0, 1]`` for a vector hit.
            pg_uri: PostgreSQL URI with pgvector; defaults to settings.
            redis_uri: Redis URI for TTL response cache. Empty or whitespace-only
                disables Redis (Postgres only).
            embedder: Custom embedder; defaults to ``get_embedder(settings)``.
            embedding_dim: When set, must equal ``embedder.embedding_dim`` (safety
                check). The embedder defines the vector width and storage namespace.
            settings: Base settings object; defaults to ``get_cache_settings()``.
        """
        self._settings = settings if settings is not None else get_cache_settings()
        self.threshold = (
            threshold if threshold is not None else self._settings.threshold
        )
        self.pg_uri = pg_uri if pg_uri is not None else self._settings.pg_uri
        resolved_redis = (
            redis_uri if redis_uri is not None else self._settings.redis_uri
        )
        self.redis_uri = resolved_redis
        self._embedder = (
            embedder if embedder is not None else get_embedder(self._settings)
        )
        resolved_dim = self._embedder.embedding_dim
        if embedding_dim is not None and embedding_dim != resolved_dim:
            msg = (
                f"embedding_dim={embedding_dim} does not match "
                f"embedder.embedding_dim={resolved_dim}"
            )
            raise ValueError(msg)
        self._embedding_dim = resolved_dim
        table_name, redis_prefix = embedding_storage_ids(
            self._embedder.cache_namespace,
            self._embedding_dim,
        )
        self._redis_key_prefix = redis_prefix
        max_pg = self._settings.pg_pool_size + self._settings.pg_pool_max_overflow
        self._vector_store = AsyncPgVectorStore(
            self.pg_uri,
            table_name=table_name,
            embedding_dim=self._embedding_dim,
            min_pool_size=self._settings.pg_pool_size,
            max_pool_size=max_pg,
        )
        self._redis_store: RedisResponseStore | None = (
            RedisResponseStore(
                resolved_redis.strip(),
                default_ttl_seconds=self._settings.redis_ttl_seconds,
            )
            if resolved_redis.strip()
            else None
        )
        self._pg_open = False
        self._closed = False
        self._embed_timeout_seconds = self._settings.embed_timeout_seconds
        self._store_timeout_seconds = self._settings.store_timeout_seconds
        self._timeout_counts = Counter()

    @property
    def timeout_counts(self) -> dict[str, int]:
        """Return observed timeout counts by operation label."""
        return dict(self._timeout_counts)

    async def _with_timeout(
        self,
        *,
        operation: str,
        timeout_seconds: float | None,
        work: Awaitable[_T],
    ) -> _T:
        """Run ``work`` with optional timeout and uniform timeout errors.

        Args:
            operation: Operation label used in logs and metrics.
            timeout_seconds: Timeout budget in seconds; disabled when ``None``.
            work: Awaitable to execute.

        Returns:
            The awaited result from ``work``.

        Raises:
            CacheTimeoutError: If the timeout expires before completion.
        """
        if timeout_seconds is None:
            return await work
        try:
            return await asyncio.wait_for(work, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            self._timeout_counts[operation] += 1
            _logger.warning(
                "Semantic cache operation timed out: operation=%s timeout_seconds=%.3f",
                operation,
                timeout_seconds,
            )
            raise CacheTimeoutError(
                operation=operation,
                timeout_seconds=timeout_seconds,
            ) from exc

    async def _ensure_open(self) -> None:
        """Open the pgvector pool on first use."""
        if self._closed:
            msg = "SemanticCache is closed"
            raise RuntimeError(msg)
        if not self._pg_open:
            await self._with_timeout(
                operation="db_open",
                timeout_seconds=self._store_timeout_seconds,
                work=self._vector_store.open(),
            )
            await self._with_timeout(
                operation="db_ensure_schema",
                timeout_seconds=self._store_timeout_seconds,
                work=self._vector_store.ensure_schema(),
            )
            self._pg_open = True

    async def get(self, query: str, model: str | None = None) -> CacheResult:
        """Embed the query, search vectors, then optionally resolve Redis by row id.

        On a vector hit, the response body prefers Redis (keyed by embedder namespace
        plus row id) when present; otherwise the JSON from Postgres is returned.

        Args:
            query: Text to embed and match semantically.
            model: Reserved for future embedder routing (unused).

        Returns:
            ``CacheResult`` with ``is_hit`` False on vector miss, else True with
            similarity and response payload.
        """
        _ = model
        src = _embed_source(self._settings)
        await self._ensure_open()
        vectors = await self._with_timeout(
            operation="embed_get",
            timeout_seconds=self._embed_timeout_seconds,
            work=self._embedder.embed([query]),
        )
        if not vectors:
            return CacheResult(is_hit=False, similarity=None, source=src, response=None)
        query_embedding = vectors[0]
        # Stage 1: fetch top-k nearest candidates using the primary threshold.
        top_k = max(1, getattr(self._settings, "top_k_candidates", 1))
        entries = await self._with_timeout(
            operation="db_similarity_search",
            timeout_seconds=self._store_timeout_seconds,
            work=self._vector_store.similarity_search_top_k(
                query_embedding=query_embedding,
                threshold=self.threshold,
                limit=top_k,
            ),
        )
        if not entries:
            return CacheResult(is_hit=False, similarity=None, source=src, response=None)

        # Stage 2: apply optional rejection threshold to filter borderline scores.
        rejection_threshold = getattr(self._settings, "rejection_threshold", None)
        chosen_entry = None
        if rejection_threshold is not None:
            for candidate in entries:
                if candidate.similarity >= rejection_threshold:
                    chosen_entry = candidate
                    break
            if chosen_entry is None:
                # All candidates failed the stricter second-stage gate; treat as miss.
                return CacheResult(
                    is_hit=False, similarity=None, source=src, response=None
                )
        else:
            chosen_entry = entries[0]

        response: dict[str, object] = chosen_entry.response
        if self._redis_store is not None:
            from_redis = await self._with_timeout(
                operation="redis_get",
                timeout_seconds=self._store_timeout_seconds,
                work=self._redis_store.get(
                    f"{self._redis_key_prefix}:{chosen_entry.id}"
                ),
            )
            if from_redis is not None:
                response = from_redis
        return CacheResult(
            is_hit=True,
            similarity=chosen_entry.similarity,
            source=src,
            response=response,
        )

    async def put(
        self, query: str, response: dict[str, object], model: str | None = None
    ) -> None:
        """Embed and persist the response in Postgres and optionally Redis.

        Redis stores the same payload under a key scoped to this embedder plus
        ``row_id``, with configured TTL.

        Args:
            query: Query text stored alongside the embedding in the pgvector table.
            response: JSON-serializable payload (mirrored in Redis when enabled).
            model: Reserved for future embedder routing (unused).
        """
        _ = model
        await self._ensure_open()
        vectors = await self._with_timeout(
            operation="embed_put",
            timeout_seconds=self._embed_timeout_seconds,
            work=self._embedder.embed([query]),
        )
        if not vectors:
            msg = "embedder returned no vectors for a non-empty query list"
            raise RuntimeError(msg)
        embedding = vectors[0]
        row_id = await self._with_timeout(
            operation="db_upsert",
            timeout_seconds=self._store_timeout_seconds,
            work=self._vector_store.upsert(query, embedding, response),
        )
        if self._redis_store is not None:
            await self._with_timeout(
                operation="redis_put",
                timeout_seconds=self._store_timeout_seconds,
                work=self._redis_store.put(
                    f"{self._redis_key_prefix}:{row_id}",
                    response,
                ),
            )

    async def close(self) -> None:
        """Close the Postgres pool and Redis client if they were opened."""
        if self._closed:
            return
        self._closed = True
        if self._pg_open:
            await self._vector_store.close()
            self._pg_open = False
        if self._redis_store is not None:
            await self._redis_store.close()
