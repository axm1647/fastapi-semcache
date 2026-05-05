"""Tests for ``SemanticCache`` cache hit/miss paths and custom embedders."""

# Tests inject mocks on ``SemanticCache`` internals for isolation.
# pyright: reportPrivateUsage=false
# pyright: reportCallIssue=false

from __future__ import annotations

from typing import override
from unittest.mock import AsyncMock

import pytest

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
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


def test_embedding_dim_mismatch_raises() -> None:
    """Passing a mismatched ``embedding_dim`` surfaces a ``ValueError``."""
    emb = _FixedEmbedder(dim=8)
    with pytest.raises(ValueError, match="embedding_dim"):
        SemanticCache(
            embedder=emb,
            embedding_dim=4,
            pg_uri="postgresql://mock/mock",
            redis_uri="",
            settings=CacheSettings(),
        )


def _make_cache(
    embedder: BaseEmbedder, *, settings: CacheSettings | None = None
) -> SemanticCache:
    """Build a cache with Redis disabled and the given embedder."""
    s = (
        settings
        if settings is not None
        else CacheSettings(redis_uri=" ", pg_uri="postgresql://mock/mock")
    )
    return SemanticCache(
        embedder=embedder,
        pg_uri="postgresql://mock/mock",
        redis_uri="",
        settings=s,
    )


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
async def test_get_hit_prefers_redis_when_enabled() -> None:
    """Redis payload replaces Postgres JSON when present."""
    settings = CacheSettings(
        redis_uri="redis://localhost:6379/0",
        pg_uri="postgresql://mock/mock",
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
    mock_redis.get = AsyncMock(return_value={"from": "redis"})
    cache._redis_store = mock_redis

    result = await cache.get("hello")
    assert result.is_hit is True
    assert result.response == {"from": "redis"}
    mock_redis.get.assert_awaited_once()


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
