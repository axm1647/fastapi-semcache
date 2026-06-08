"""Tests for ``SemanticCache`` cache hit/miss paths and custom embedders."""

# Tests inject mocks on ``SemanticCache`` internals for isolation.
# pyright: reportPrivateUsage=false
# pyright: reportCallIssue=false

from __future__ import annotations

import asyncio
from typing import override
from unittest.mock import AsyncMock

import pytest

from semanticcache.cache import SemanticCache, resolve_cache_scope
from semanticcache.config import CacheSettings
from semanticcache.exceptions import CacheTimeoutError
from semanticcache.embedders import BaseEmbedder
from semanticcache.types import CacheEntry, CacheResult


class _FixedEmbedder(BaseEmbedder):
    """Deterministic embedder for unit tests (fixed width, repeat character encodings)."""

    def __init__(self, *, dim: int = 4) -> None:
        self._dim = dim

    @property
    @override
    def embedding_dim(self) -> int:
        return self._dim

    @property
    @override
    def cache_namespace(self) -> str:
        return f"test:fixed:{self._dim}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vec = [float(i + 1) / float(self._dim) for i in range(self._dim)]
        return [list(vec) for _ in texts]


class _EmptyEmbedder(_FixedEmbedder):
    """Always returns no vectors (invalid but useful for error-path tests)."""

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        return []


class _SlowEmbedder(_FixedEmbedder):
    """Sleep before returning vectors to simulate a slow provider."""

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0.05)
        return await super().embed(texts)


class _CountingEmbedder(_FixedEmbedder):
    """Track embed calls to assert miss-path vector reuse."""

    def __init__(self, *, dim: int = 4) -> None:
        super().__init__(dim=dim)
        self.calls = 0

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return await super().embed(texts)


def test_embedding_dim_mismatch_raises() -> None:
    """Passing a mismatched ``embedding_dim`` surfaces a ``ValueError``."""
    emb = _FixedEmbedder(dim=8)
    with pytest.raises(ValueError, match="embedding_dim"):
        SemanticCache(
            embedder=emb,
            embedding_dim=4,
            pg_uri="postgresql://mock/mock",
            redis_uri="",
            settings=CacheSettings(require_cache_scope=False),
        )


def test_semantic_cache_requires_embedder_when_custom_default() -> None:
    """Bare SemanticCache() fails when embedder_type is custom (the default)."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
    )
    with pytest.raises(ValueError, match="pass embedder="):
        SemanticCache(
            pg_uri="postgresql://mock/mock",
            redis_uri="",
            settings=settings,
        )


def test_semantic_cache_accepts_custom_embedder_with_custom_type() -> None:
    """Custom embedder= works when embedder_type remains custom (default)."""
    emb = _FixedEmbedder(dim=4)
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
        embedder_type="custom",
    )
    cache = SemanticCache(
        embedder=emb,
        pg_uri="postgresql://mock/mock",
        redis_uri="",
        settings=settings,
    )
    assert cache._embedder is emb


def _make_cache(
    embedder: BaseEmbedder, *, settings: CacheSettings | None = None
) -> SemanticCache:
    """Build a cache with Redis disabled and the given embedder."""
    s = (
        settings
        if settings is not None
        else CacheSettings(
            redis_uri=" ",
            pg_uri="postgresql://mock/mock",
            require_cache_scope=False,
        )
    )
    return SemanticCache(
        embedder=embedder,
        pg_uri="postgresql://mock/mock",
        redis_uri="",
        settings=s,
    )


def test_cache_settings_configure_pgvector_hnsw_defaults() -> None:
    """SemanticCache forwards pgvector HNSW defaults into the vector store."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
        pgvector_hnsw_m=24,
        pgvector_hnsw_ef_construction=96,
        pgvector_hnsw_ef_search=80,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)

    assert cache._vector_store._hnsw_m == 24
    assert cache._vector_store._hnsw_ef_construction == 96
    assert cache._vector_store._default_hnsw_ef_search == 80


@pytest.mark.asyncio
async def test_ensure_open_skips_schema_when_pg_ensure_schema_false() -> None:
    """When runtime DDL is disabled, ``ensure_schema`` is not invoked on first use."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
        pg_ensure_schema=False,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    await cache.get("hello world")

    mock_vs.open.assert_awaited_once()
    mock_vs.ensure_schema.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_miss_when_store_returns_none() -> None:
    """Vector miss yields ``is_hit`` False and empty payload."""
    cache = _make_cache(_FixedEmbedder())
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    result = await cache.get("hello world")
    assert result.is_hit is False
    assert result.similarity is None
    assert result.response is None
    mock_vs.similarity_search_top_k.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_hit_returns_redis_free_payload() -> None:
    """Vector hit returns Postgres payload when Redis is disabled."""
    cache = _make_cache(_FixedEmbedder())
    entry = CacheEntry(
        id=1,
        query_text="stored query",
        response={"answer": 42},
        similarity=0.95,
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[entry])
    cache._vector_store = mock_vs

    result = await cache.get("similar query")
    assert result.is_hit is True
    assert result.similarity is not None and abs(result.similarity - 0.95) < 1e-9
    assert result.response == {"answer": 42}


@pytest.mark.asyncio
async def test_get_passes_hnsw_ef_search_override_to_vector_store() -> None:
    """Per-call HNSW ef_search override is forwarded to pgvector search."""
    cache = _make_cache(_FixedEmbedder())
    entry = CacheEntry(
        id=1,
        query_text="stored query",
        response={"answer": 42},
        similarity=0.95,
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[entry])
    cache._vector_store = mock_vs

    result = await cache.get("similar query", hnsw_ef_search=120)

    assert result.is_hit is True
    assert mock_vs.similarity_search_top_k.await_args.kwargs["ef_search"] == 120


@pytest.mark.asyncio
async def test_get_hit_prefers_redis_when_enabled() -> None:
    """Redis payload replaces Postgres JSON when present (semantic hit path)."""
    settings = CacheSettings(
        redis_uri="redis://localhost:6379/0",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)

    entry = CacheEntry(
        id=7,
        query_text="q",
        response={"from": "postgres"},
        similarity=0.91,
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[entry])
    cache._vector_store = mock_vs

    mock_redis = AsyncMock()
    # Return None for exact-match lookup so the semantic path is exercised,
    # then return the response blob for the response-key lookup.
    mock_redis.get = AsyncMock(
        side_effect=lambda key: None if ":exact:" in key else {"from": "redis"}
    )
    cache._redis_store = mock_redis

    result = await cache.get("hello")
    assert result.is_hit is True
    assert result.response == {"from": "redis"}
    all_keys = [call[0][0] for call in mock_redis.get.await_args_list]
    response_keys = [k for k in all_keys if ":exact:" not in k]
    assert len(response_keys) == 1
    assert response_keys[0].endswith(":default:default:7")


@pytest.mark.asyncio
async def test_get_rejection_threshold_filters_out_borderline_candidates() -> None:
    """Second-stage rejection threshold can turn candidates into a miss."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        threshold=0.80,
        top_k_candidates=3,
        rejection_threshold=0.90,
        require_cache_scope=False,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)

    # All candidates are above the primary threshold but below the stricter rejection
    # threshold, so the second stage should yield a miss.
    entries = [
        CacheEntry(
            id=1,
            query_text="q1",
            response={"answer": 1},
            similarity=0.85,
        ),
        CacheEntry(
            id=2,
            query_text="q2",
            response={"answer": 2},
            similarity=0.88,
        ),
    ]

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=entries)
    cache._vector_store = mock_vs

    result = await cache.get("borderline query")
    assert result.is_hit is False
    assert result.response is None
    assert result.similarity is None
    mock_vs.similarity_search_top_k.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_rejection_threshold_accepts_strong_candidate() -> None:
    """Second-stage rejection threshold selects the first strong-enough candidate."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        threshold=0.80,
        top_k_candidates=3,
        rejection_threshold=0.90,
        require_cache_scope=False,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)

    weak = CacheEntry(
        id=1,
        query_text="weak",
        response={"answer": "weak"},
        similarity=0.88,
    )
    strong = CacheEntry(
        id=2,
        query_text="strong",
        response={"answer": "strong"},
        similarity=0.93,
    )

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[weak, strong])
    cache._vector_store = mock_vs

    result = await cache.get("query")
    assert result.is_hit is True
    assert result.response == {"answer": "strong"}
    assert result.similarity is not None and abs(result.similarity - 0.93) < 1e-9
    mock_vs.similarity_search_top_k.assert_awaited_once()


@pytest.mark.asyncio
async def test_put_raises_when_embed_returns_empty() -> None:
    """Store path fails closed when ``embed`` yields no rows."""
    cache = _make_cache(_EmptyEmbedder())

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    cache._vector_store = mock_vs

    with pytest.raises(RuntimeError, match="no vectors"):
        await cache.put("x", {"a": 1})


@pytest.mark.asyncio
async def test_put_persists_via_vector_store() -> None:
    """``put`` calls ``upsert`` with an embedding row aligned to the query."""
    cache = _make_cache(_FixedEmbedder())
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.upsert = AsyncMock(return_value=3)
    cache._vector_store = mock_vs

    await cache.put("abc", {"ok": True})
    mock_vs.upsert.assert_awaited_once()
    call = mock_vs.upsert.await_args
    assert call[0][0] == "abc"
    assert len(call[0][1]) == 4
    assert call[0][2] == {"ok": True}
    assert call.kwargs.get("model_key") == ""


@pytest.mark.asyncio
async def test_get_empty_embed_returns_miss() -> None:
    """If ``embed`` returns an empty list, treat as miss (defensive)."""
    cache = _make_cache(_EmptyEmbedder())

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    cache._vector_store = mock_vs

    out = await cache.get("q")
    assert out == CacheResult(
        is_hit=False, similarity=None, source="embedders.sbert", response=None
    )
    mock_vs.similarity_search_top_k.assert_not_called()


@pytest.mark.asyncio
async def test_get_raises_timeout_when_embedder_is_slow() -> None:
    """Embedder slowness raises a uniform timeout exception."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        embed_timeout_seconds=0.01,
        store_timeout_seconds=1.0,
        require_cache_scope=False,
    )
    cache = _make_cache(_SlowEmbedder(), settings=settings)

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    cache._vector_store = mock_vs

    with pytest.raises(CacheTimeoutError, match="embed_get"):
        await cache.get("too slow")
    assert cache.timeout_counts.get("embed_get") == 1
    mock_vs.similarity_search_top_k.assert_not_called()


@pytest.mark.asyncio
async def test_get_put_scope_by_model_key() -> None:
    """Different ``model`` values use distinct vector search keys and Redis paths."""
    settings = CacheSettings(
        redis_uri="redis://localhost:6379/0",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)

    entry_a = CacheEntry(
        id=1,
        query_text="q",
        response={"which": "a"},
        similarity=0.95,
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[entry_a])
    mock_vs.upsert = AsyncMock(return_value=99)
    cache._vector_store = mock_vs

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    cache._redis_store = mock_redis

    await cache.get("hello", model="gpt-4")
    await cache.put("hello", {"x": 1}, model="claude-3")

    assert mock_vs.similarity_search_top_k.await_args.kwargs["model_key"] == "gpt-4"
    assert mock_vs.similarity_search_top_k.await_args.kwargs["scope_key"] == ""
    assert mock_vs.upsert.await_args.kwargs["model_key"] == "claude-3"
    assert mock_vs.upsert.await_args.kwargs["scope_key"] == ""


@pytest.mark.asyncio
async def test_normalize_model_none_and_whitespace_use_default_bucket() -> None:
    """``None`` and blank ``model`` share the default ``model_key``."""
    cache = _make_cache(_FixedEmbedder())
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    await cache.get("q", model=None)
    await cache.get("q", model="   ")
    assert mock_vs.similarity_search_top_k.await_args_list[0].kwargs["model_key"] == ""
    assert mock_vs.similarity_search_top_k.await_args_list[1].kwargs["model_key"] == ""


@pytest.mark.asyncio
async def test_put_raises_timeout_when_store_is_slow() -> None:
    """Store slowness raises a uniform timeout exception."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        embed_timeout_seconds=1.0,
        store_timeout_seconds=0.01,
        require_cache_scope=False,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()

    async def _slow_upsert(*args: object, **kwargs: object) -> int:
        _ = args, kwargs
        await asyncio.sleep(0.05)
        return 1

    mock_vs.upsert = AsyncMock(side_effect=_slow_upsert)
    cache._vector_store = mock_vs

    with pytest.raises(CacheTimeoutError, match="db_upsert"):
        await cache.put("abc", {"ok": True})
    assert cache.timeout_counts.get("db_upsert") == 1


def test_resolve_cache_scope_required_rejects_empty() -> None:
    """When scope is mandatory, missing or blank values resolve to cache bypass."""
    settings = CacheSettings(require_cache_scope=True)
    assert resolve_cache_scope(None, settings=settings) is None
    assert resolve_cache_scope("", settings=settings) is None
    assert resolve_cache_scope("   ", settings=settings) is None


def test_resolve_cache_scope_required_accepts_value() -> None:
    """Required scope mode normalizes non-empty strings."""
    settings = CacheSettings(require_cache_scope=True)
    assert resolve_cache_scope("  t1  ", settings=settings) == "t1"


def test_resolve_cache_scope_optional_allows_empty_bucket() -> None:
    """Legacy single-tenant mode maps missing scope to the shared empty bucket."""
    settings = CacheSettings(require_cache_scope=False)
    assert resolve_cache_scope(None, settings=settings) == ""
    assert resolve_cache_scope("", settings=settings) == ""


@pytest.mark.asyncio
async def test_get_prefers_storage_scope_key_over_scope_argument() -> None:
    """``storage_scope_key`` skips re-resolution of ``scope`` (middleware fast path)."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=True,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)
    entry = CacheEntry(
        id=1,
        query_text="q",
        response={"bucket": "scoped"},
        similarity=0.95,
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[entry])
    cache._vector_store = mock_vs

    out = await cache.get(
        "hello",
        scope=None,
        storage_scope_key="tenant-a",
    )
    assert out.is_hit is True
    assert mock_vs.similarity_search_top_k.await_args.kwargs["scope_key"] == "tenant-a"


@pytest.mark.asyncio
async def test_get_bypasses_store_when_required_scope_missing() -> None:
    """With ``require_cache_scope`` True, a missing scope avoids embedding and IO."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=True,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)
    mock_vs = AsyncMock()
    cache._vector_store = mock_vs

    result = await cache.get("hello")
    assert result.is_hit is False
    mock_vs.open.assert_not_called()
    mock_vs.similarity_search_top_k.assert_not_called()


@pytest.mark.asyncio
async def test_put_noops_when_required_scope_missing() -> None:
    """With ``require_cache_scope`` True, ``put`` without scope does not persist."""
    settings = CacheSettings(
        redis_uri=" ",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=True,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)
    mock_vs = AsyncMock()
    cache._vector_store = mock_vs

    await cache.put("abc", {"ok": True})
    mock_vs.open.assert_not_called()
    mock_vs.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_get_put_pass_scope_key_when_required() -> None:
    """Non-empty scope is forwarded to the vector store and Redis key layout."""
    settings = CacheSettings(
        redis_uri="redis://localhost:6379/0",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=True,
    )
    cache = _make_cache(_FixedEmbedder(), settings=settings)
    entry = CacheEntry(
        id=3,
        query_text="q",
        response={"k": 1},
        similarity=0.92,
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[entry])
    mock_vs.upsert = AsyncMock(return_value=4)
    cache._vector_store = mock_vs
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    cache._redis_store = mock_redis

    await cache.get("hello", scope="org-9")
    await cache.put("hello", {"x": 1}, scope="org-9")

    assert mock_vs.similarity_search_top_k.await_args.kwargs["scope_key"] == "org-9"
    assert mock_vs.upsert.await_args.kwargs["scope_key"] == "org-9"
    get_key = mock_redis.get.await_args[0][0]
    assert get_key.endswith(":3")
    # put is called twice: response blob first, then exact-match id entry.
    put_keys = [call[0][0] for call in mock_redis.put.await_args_list]
    response_put_keys = [k for k in put_keys if ":exact:" not in k]
    assert len(response_put_keys) == 1
    assert response_put_keys[0].endswith(":4")


@pytest.mark.asyncio
async def test_put_reuses_query_embedding_from_get_miss() -> None:
    """Miss lookup vector can be reused to avoid a second embedding call."""
    embedder = _CountingEmbedder()
    cache = _make_cache(embedder)
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    mock_vs.upsert = AsyncMock(return_value=11)
    cache._vector_store = mock_vs

    miss = await cache.get("reuse me")
    assert miss.is_hit is False
    assert miss.query_embedding is not None
    assert embedder.calls == 1

    await cache.put("reuse me", {"ok": True}, query_embedding=miss.query_embedding)
    assert embedder.calls == 1
    mock_vs.upsert.assert_awaited_once()


class _EmbedderWithAclose(_FixedEmbedder):
    """Fixed embedder that tracks ``aclose`` for shutdown tests."""

    def __init__(self, *, dim: int = 4) -> None:
        super().__init__(dim=dim)
        self.aclose = AsyncMock()


@pytest.mark.asyncio
async def test_close_invokes_embedder_aclose_when_present() -> None:
    """``SemanticCache.close`` awaits ``embedder.aclose`` when implemented."""
    embedder = _EmbedderWithAclose()
    cache = _make_cache(embedder)
    cache._vector_store.close = AsyncMock()
    cache._pg_open = True

    await cache.close()

    embedder.aclose.assert_awaited_once()
    assert cache._closed is True


@pytest.mark.asyncio
async def test_close_is_idempotent_for_embedder_aclose() -> None:
    """Second ``close`` does not call ``aclose`` again."""
    embedder = _EmbedderWithAclose()
    cache = _make_cache(embedder)
    cache._vector_store.close = AsyncMock()
    cache._pg_open = True

    await cache.close()
    await cache.close()

    embedder.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Exact-match Redis fast path tests
# ---------------------------------------------------------------------------


def _make_cache_with_redis(embedder: BaseEmbedder) -> SemanticCache:
    """Build a cache with Redis enabled and the given embedder."""
    settings = CacheSettings(
        redis_uri="redis://localhost:6379/0",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=False,
    )
    return SemanticCache(
        embedder=embedder,
        pg_uri="postgresql://mock/mock",
        redis_uri="redis://localhost:6379/0",
        settings=settings,
    )


@pytest.mark.asyncio
async def test_exact_match_redis_hit_skips_embedding() -> None:
    """Exact-text Redis hit returns response without calling the embedder."""
    embedder = _CountingEmbedder()
    cache = _make_cache_with_redis(embedder)

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    cache._vector_store = mock_vs

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(
        side_effect=lambda key: (
            {"id": 42} if ":exact:" in key else {"from": "redis"}
        )
    )
    cache._redis_store = mock_redis

    result = await cache.get("exact query text")

    assert result.is_hit is True
    assert result.response == {"from": "redis"}
    assert result.similarity == 1.0
    assert result.cache_entry_id == 42
    assert embedder.calls == 0
    mock_vs.similarity_search_top_k.assert_not_called()


@pytest.mark.asyncio
async def test_exact_match_redis_miss_falls_through_to_embedding() -> None:
    """When exact-match Redis key is absent, embedding and ANN search run normally."""
    embedder = _CountingEmbedder()
    cache = _make_cache_with_redis(embedder)

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    cache._redis_store = mock_redis

    result = await cache.get("unseen query")

    assert result.is_hit is False
    assert embedder.calls == 1
    mock_vs.similarity_search_top_k.assert_awaited_once()


@pytest.mark.asyncio
async def test_exact_match_redis_blob_expired_falls_back_to_postgres() -> None:
    """When exact key is present but response blob is gone, Postgres is the fallback."""
    embedder = _CountingEmbedder()
    cache = _make_cache_with_redis(embedder)

    pg_entry = CacheEntry(
        id=7,
        query_text="stored query",
        response={"from": "postgres"},
        similarity=1.0,
    )

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.get_by_id = AsyncMock(return_value=pg_entry)
    cache._vector_store = mock_vs

    def _redis_get(key: str) -> dict | None:
        if ":exact:" in key:
            return {"id": 7}
        return None  # response blob evicted

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=_redis_get)
    cache._redis_store = mock_redis

    result = await cache.get("some query")

    assert result.is_hit is True
    assert result.response == {"from": "postgres"}
    assert result.similarity == 1.0
    assert result.cache_entry_id == 7
    assert embedder.calls == 0
    mock_vs.get_by_id.assert_awaited_once_with(
        7, model_key="", scope_key=""
    )
    mock_vs.similarity_search_top_k.assert_not_called()


@pytest.mark.asyncio
async def test_exact_match_redis_postgres_fallback_also_misses() -> None:
    """When both Redis blob and Postgres row are gone, falls through to ANN search."""
    embedder = _CountingEmbedder()
    cache = _make_cache_with_redis(embedder)

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.get_by_id = AsyncMock(return_value=None)
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    def _redis_get(key: str) -> dict | None:
        if ":exact:" in key:
            return {"id": 99}
        return None

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=_redis_get)
    cache._redis_store = mock_redis

    result = await cache.get("stale query")

    assert result.is_hit is False
    assert embedder.calls == 1
    mock_vs.get_by_id.assert_awaited_once()
    mock_vs.similarity_search_top_k.assert_awaited_once()


@pytest.mark.asyncio
async def test_put_writes_exact_match_key_to_redis() -> None:
    """``put`` writes both the response blob and the exact-match id entry to Redis."""
    embedder = _FixedEmbedder()
    cache = _make_cache_with_redis(embedder)

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.upsert = AsyncMock(return_value=55)
    cache._vector_store = mock_vs

    mock_redis = AsyncMock()
    cache._redis_store = mock_redis

    await cache.put("store this query", {"answer": "yes"})

    assert mock_redis.put.await_count == 2
    call_keys = [call[0][0] for call in mock_redis.put.await_args_list]
    assert any(":exact:" in k for k in call_keys), (
        f"Expected one exact-match key among {call_keys}"
    )
    assert any(":exact:" not in k for k in call_keys), (
        f"Expected one response blob key among {call_keys}"
    )
    exact_call = next(c for c in mock_redis.put.await_args_list if ":exact:" in c[0][0])
    assert exact_call[0][1] == {"id": 55}


@pytest.mark.asyncio
async def test_exact_match_key_is_scoped_by_model_and_scope() -> None:
    """Exact-match keys include hashed model and scope segments."""
    settings = CacheSettings(
        redis_uri="redis://localhost:6379/0",
        pg_uri="postgresql://mock/mock",
        require_cache_scope=True,
    )
    cache = SemanticCache(
        embedder=_FixedEmbedder(),
        pg_uri="postgresql://mock/mock",
        redis_uri="redis://localhost:6379/0",
        settings=settings,
    )

    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.upsert = AsyncMock(return_value=10)
    cache._vector_store = mock_vs

    mock_redis = AsyncMock()
    cache._redis_store = mock_redis

    await cache.put("q", {"x": 1}, model="gpt-4", scope="org-1")
    await cache.put("q", {"x": 1}, model="gpt-4", scope="org-2")

    exact_keys = [
        call[0][0]
        for call in mock_redis.put.await_args_list
        if ":exact:" in call[0][0]
    ]
    assert len(exact_keys) == 2
    assert exact_keys[0] != exact_keys[1], (
        "Exact-match keys must differ across scope buckets"
    )
