"""High-level semantic cache orchestrating embedder, pgvector, and optional Redis."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import Counter
from typing import TYPE_CHECKING, Awaitable, TypeVar

from .config import get_cache_settings
from .embedders import BaseEmbedder, get_embedder
from .exceptions import CacheTimeoutError
from .stores import AsyncPgVectorStore, RedisResponseStore
from .stores.vector.storage_ids import embedding_storage_ids
from .types import CacheResult, CacheSource

if TYPE_CHECKING:
    from semanticcache.config import CacheSettings

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _normalize_model_key(model: str | None) -> str:
    """Normalize optional LLM or routing model id for storage lookup.

    Args:
        model: Caller-supplied model id, or ``None`` when omitted.

    Returns:
        Stripped UTF-8 string; ``None`` and whitespace-only values become ``""``
        (default bucket shared with explicit empty string).
    """
    if model is None:
        return ""
    return model.strip()


def _redis_bucket_segment(raw_key: str) -> str:
    """Return a short stable Redis key segment for a scope or model bucket string.

    Args:
        raw_key: Normalized ``scope_key`` or ``model_key`` (possibly empty).

    Returns:
        The literal ``default`` for the empty bucket, else the first 16 hex chars of
        the SHA-256 digest of the UTF-8 key (collision-safe for cache routing).
    """
    if not raw_key:
        return "default"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def _resolve_scope_key_for_storage(
    *,
    scope: str | None,
    storage_scope_key: str | None,
    settings: "CacheSettings",
) -> str | None:
    """Produce ``scope_key`` for Postgres and Redis from raw or pre-resolved input.

    When ``storage_scope_key`` is set (typically by ``SemanticCacheMiddleware`` after a
    single ``resolve_cache_scope`` call), it is used directly after trimming so callers
    avoid resolving twice. When unset, ``scope`` is passed through ``resolve_cache_scope``.

    Args:
        scope: Raw tenant or namespace string from the caller or extractor.
        storage_scope_key: Optional key already normalized per ``resolve_cache_scope``;
            takes precedence over ``scope`` when not ``None``.
        settings: Active cache settings.

    Returns:
        Storage scope string, or ``None`` when the cache must not read or write.
    """
    if storage_scope_key is not None:
        trimmed = storage_scope_key.strip()
        if settings.require_cache_scope:
            return trimmed if trimmed else None
        return trimmed
    return resolve_cache_scope(scope, settings=settings)


def resolve_cache_scope(raw: str | None, *, settings: "CacheSettings") -> str | None:
    """Resolve caller-provided scope into storage form or signal cache bypass.

    When ``settings.require_cache_scope`` is True, empty or missing scope means the
    caller must not read or write the shared cache (prevents cross-tenant hits).

    Args:
        raw: Tenant or namespace string, or ``None`` when the caller did not supply
            one.
        settings: Active cache settings (``require_cache_scope`` governs behavior).

    Returns:
        Normalized non-empty scope string for Postgres and Redis, ``""`` when
        ``require_cache_scope`` is False and scope is optional (legacy single-tenant
        bucket), or ``None`` when cache operations must be skipped entirely.
    """
    stripped = "" if raw is None else raw.strip()
    if settings.require_cache_scope:
        if not stripped:
            return None
        return stripped
    return stripped


def _embed_source(
    settings: "CacheSettings",
) -> CacheSource:
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
    _open_lock: asyncio.Lock
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
        self._open_lock = asyncio.Lock()
        self._embed_timeout_seconds = self._settings.embed_timeout_seconds
        self._store_timeout_seconds = self._settings.store_timeout_seconds
        self._timeout_counts = Counter()
        if self._embed_timeout_seconds is None:
            _logger.warning(
                "embed_timeout_seconds is None, embedder calls will not be timed out"
            )
        if self._store_timeout_seconds is None:
            _logger.warning(
                "store_timeout_seconds is None, database operations will not be timed "
                "out"
            )

    @property
    def settings(self) -> "CacheSettings":
        """Return the cache settings used by this instance."""
        return self._settings

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
        """Open the pgvector pool on first use.

        Uses a double-checked lock so that concurrent callers on the same
        event-loop iteration each wait for the single initialization path
        rather than racing to open the pool multiple times.
        """
        if self._closed:
            msg = "SemanticCache is closed"
            raise RuntimeError(msg)
        if self._pg_open:
            return
        async with self._open_lock:
            if self._pg_open:
                return
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

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
        """Embed the query, search vectors, then optionally resolve Redis by row id.

        On a vector hit, the response body prefers Redis (keyed by embedder namespace,
        scope bucket, model bucket, and row id) when present; otherwise the JSON from
        Postgres is returned.

        Args:
            query: Text to embed and match semantically.
            model: Logical model id for scoped lookup (same value as ``put``); ``None``
                or whitespace-only values use the default bucket (``model_key=""``).
            scope: Tenant or namespace id for isolation; must match ``put`` when
                ``require_cache_scope`` is True. When resolution yields no scope in that
                mode, this method returns a miss without embedding. Ignored when
                ``storage_scope_key`` is not ``None``.
            storage_scope_key: Optional key already produced by ``resolve_cache_scope``
                for this request; middleware passes this to avoid resolving twice. Call
                sites that set this should treat it as internal coordination with the
                same ``SemanticCache.settings`` used here.

        Returns:
            ``CacheResult`` with ``is_hit`` False on vector miss, else True with
            similarity and response payload. Miss results may include
            ``query_embedding`` so callers can reuse it in a follow-up ``put``.
        """
        model_key = _normalize_model_key(model)
        scope_key = _resolve_scope_key_for_storage(
            scope=scope,
            storage_scope_key=storage_scope_key,
            settings=self._settings,
        )
        src = _embed_source(self._settings)
        if scope_key is None:
            return CacheResult(is_hit=False, similarity=None, source=src, response=None)
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
                model_key=model_key,
                scope_key=scope_key,
            ),
        )
        if not entries:
            miss = CacheResult(
                is_hit=False,
                similarity=None,
                source=src,
                response=None,
            )
            miss.query_embedding = query_embedding
            return miss

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
                miss = CacheResult(
                    is_hit=False,
                    similarity=None,
                    source=src,
                    response=None,
                )
                miss.query_embedding = query_embedding
                return miss
        else:
            chosen_entry = entries[0]

        response: dict[str, object] = chosen_entry.response
        if self._redis_store is not None:
            redis_key = (
                f"{self._redis_key_prefix}:{_redis_bucket_segment(scope_key)}:"
                f"{_redis_bucket_segment(model_key)}:{chosen_entry.id}"
            )
            from_redis = await self._with_timeout(
                operation="redis_get",
                timeout_seconds=self._store_timeout_seconds,
                work=self._redis_store.get(redis_key),
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
        self,
        query: str,
        response: dict[str, object],
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> None:
        """Embed and persist the response in Postgres and optionally Redis.

        Redis stores the same payload under a key scoped to this embedder, scope and
        model buckets, and ``row_id``, with configured TTL.

        Args:
            query: Query text stored alongside the embedding in the pgvector table.
            response: JSON-serializable payload (mirrored in Redis when enabled).
            model: Logical model id for scoped storage (must match ``get``); ``None``
                or whitespace-only values use the default bucket.
            scope: Tenant or namespace id (must match ``get``) when
                ``require_cache_scope`` is True; otherwise optional. Ignored when
                ``storage_scope_key`` is not ``None``.
            storage_scope_key: Optional key from ``resolve_cache_scope`` for this
                request; same semantics as ``get``.
            query_embedding: Optional precomputed embedding for ``query``. Middleware
                can pass the vector produced during ``get`` miss evaluation to avoid
                an additional embedder call before ``upsert``.
        """
        model_key = _normalize_model_key(model)
        scope_key = _resolve_scope_key_for_storage(
            scope=scope,
            storage_scope_key=storage_scope_key,
            settings=self._settings,
        )
        if scope_key is None:
            return
        await self._ensure_open()
        if query_embedding is None:
            vectors = await self._with_timeout(
                operation="embed_put",
                timeout_seconds=self._embed_timeout_seconds,
                work=self._embedder.embed([query]),
            )
            if not vectors:
                msg = "embedder returned no vectors for a non-empty query list"
                raise RuntimeError(msg)
            embedding = vectors[0]
        else:
            if len(query_embedding) != self._embedding_dim:
                msg = (
                    f"provided query_embedding length {len(query_embedding)} does not "
                    f"match embedder.embedding_dim={self._embedding_dim}"
                )
                raise ValueError(msg)
            embedding = list(query_embedding)
        row_id = await self._with_timeout(
            operation="db_upsert",
            timeout_seconds=self._store_timeout_seconds,
            work=self._vector_store.upsert(
                query,
                embedding,
                response,
                model_key=model_key,
                scope_key=scope_key,
            ),
        )
        if self._redis_store is not None:
            redis_key = (
                f"{self._redis_key_prefix}:{_redis_bucket_segment(scope_key)}:"
                f"{_redis_bucket_segment(model_key)}:{row_id}"
            )
            await self._with_timeout(
                operation="redis_put",
                timeout_seconds=self._store_timeout_seconds,
                work=self._redis_store.put(redis_key, response),
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
