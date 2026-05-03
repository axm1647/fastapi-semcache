"""High-level semantic cache orchestrating embedder, pgvector, and optional Redis."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from semanticcache.config import get_cache_settings
from semanticcache.embedders import BaseEmbedder, get_embedder
from semanticcache.stores import AsyncPgVectorStore, RedisResponseStore
from semanticcache.types import CacheResult

if TYPE_CHECKING:
    from semanticcache.config import CacheSettings


def _embed_source(
    settings: "CacheSettings",
) -> Literal["embedders.sbert", "embedders.openai", "none"]:
    """Map configured embedder type to ``CacheResult.source``."""
    if settings.embedder_type == "local":
        return "embedders.sbert"
    if settings.embedder_type == "openai":
        return "embedders.openai"
    return "none"


class SemanticCache:
    """Orchestrate embedding, pgvector ANN search, and optional Redis response cache."""

    threshold: float
    pg_uri: str
    redis_uri: str
    _embedding_dim: int
    _settings: "CacheSettings"
    _embedder: BaseEmbedder
    _vector_store: AsyncPgVectorStore
    _pg_open: bool
    _closed: bool

    def __init__(
        self,
        threshold: float | None = None,
        pg_uri: str | None = None,
        redis_uri: str | None = None,
        *,
        embedder: BaseEmbedder | None = None,
        embedding_dim: int = 384,
        settings: "CacheSettings | None" = None,
    ) -> None:
        """Build stores and embedder from explicit args or ``CacheSettings``.

        Args:
            threshold: Minimum cosine similarity in ``[0, 1]`` for a vector hit.
            pg_uri: PostgreSQL URI with pgvector; defaults to settings.
            redis_uri: Redis URI for TTL response cache. Empty or whitespace-only
                disables Redis (Postgres only).
            embedder: Custom embedder; defaults to ``get_embedder(settings)``.
            embedding_dim: Vector dimension; must match ``cache_entries`` and the
                embedder output.
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
        self._embedding_dim = embedding_dim
        self._embedder = (
            embedder if embedder is not None else get_embedder(self._settings)
        )
        max_pg = self._settings.pg_pool_size + self._settings.pg_pool_max_overflow
        self._vector_store = AsyncPgVectorStore(
            self.pg_uri,
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

    async def _ensure_open(self) -> None:
        """Open the pgvector pool on first use."""
        if self._closed:
            msg = "SemanticCache is closed"
            raise RuntimeError(msg)
        if not self._pg_open:
            await self._vector_store.open()
            self._pg_open = True

    async def get(self, query: str, model: str | None = None) -> CacheResult:
        """Embed the query, search vectors, then optionally resolve Redis by row id.

        On a vector hit, the response body prefers Redis (key ``str(entry.id)``) when
        present; otherwise the JSON from Postgres is returned.

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
        vectors = await self._embedder.embed([query])
        if not vectors:
            return CacheResult(is_hit=False, similarity=None, source=src, response=None)
        query_embedding = vectors[0]
        entry = await self._vector_store.similarity_search(
            query_embedding, self.threshold
        )
        if entry is None:
            return CacheResult(is_hit=False, similarity=None, source=src, response=None)
        response: dict[str, object] = entry.response
        if self._redis_store is not None:
            from_redis = await self._redis_store.get(str(entry.id))
            if from_redis is not None:
                response = from_redis
        return CacheResult(
            is_hit=True,
            similarity=entry.similarity,
            source=src,
            response=response,
        )

    async def put(
        self, query: str, response: dict[str, object], model: str | None = None
    ) -> None:
        """Embed and persist the response in Postgres and optionally Redis.

        Redis stores the same payload under ``str(row_id)`` with configured TTL.

        Args:
            query: Query text stored alongside the embedding in ``cache_entries``.
            response: JSON-serializable payload (mirrored in Redis when enabled).
            model: Reserved for future embedder routing (unused).
        """
        _ = model
        await self._ensure_open()
        vectors = await self._embedder.embed([query])
        if not vectors:
            msg = "embedder returned no vectors for a non-empty query list"
            raise RuntimeError(msg)
        embedding = vectors[0]
        row_id = await self._vector_store.upsert(query, embedding, response)
        if self._redis_store is not None:
            await self._redis_store.put(str(row_id), response)

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
