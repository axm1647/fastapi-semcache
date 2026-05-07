"""Tests for tenant scope extraction and SemanticCache settings alignment."""

# pyright: reportCallIssue=false
# pyright: reportPrivateUsage=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import override
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.embedders import BaseEmbedder
from semanticcache.middleware.fastapi import SemanticCacheMiddleware


class _MiniEmbedder(BaseEmbedder):
    """Tiny deterministic embedder for middleware integration tests."""

    @property
    @override
    def embedding_dim(self) -> int:
        return 4

    @property
    @override
    def cache_namespace(self) -> str:
        return "middleware-scope-test"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vec = [0.25, 0.25, 0.25, 0.25]
        return [list(vec) for _ in texts]


def _mini_semantic_cache(*, require_scope: bool) -> SemanticCache:
    """Build a SemanticCache with optional scope requirement and Redis disabled."""
    return SemanticCache(
        embedder=_MiniEmbedder(),
        pg_uri="postgresql://mock/mock",
        redis_uri="",
        settings=CacheSettings(
            redis_uri=" ",
            pg_uri="postgresql://mock/mock",
            require_cache_scope=require_scope,
        ),
    )


@pytest.mark.asyncio
async def test_json_numeric_tenant_id_is_accepted_as_scope() -> None:
    """Integer ``tenant_id`` in JSON is normalized to a string scope."""
    cache = _mini_semantic_cache(require_scope=True)
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cache,
        cache_settings=CacheSettings(require_cache_scope=False),
    )

    with TestClient(app) as client:
        r = client.post("/v1/chat", json={"query": "hello", "tenant_id": 4242})

    assert r.status_code == 200
    mock_vs.similarity_search_top_k.assert_awaited()
    kwargs = mock_vs.similarity_search_top_k.await_args.kwargs
    assert kwargs["scope_key"] == "4242"


@pytest.mark.asyncio
async def test_middleware_scope_gate_follows_semantic_cache_settings() -> None:
    """``require_cache_scope`` on ``SemanticCache.settings`` overrides ``cache_settings``."""
    cache = _mini_semantic_cache(require_scope=False)
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cache,
        cache_settings=CacheSettings(require_cache_scope=True),
    )

    with TestClient(app) as client:
        r = client.post("/v1/chat", json={"query": "only-query-no-scope"})

    assert r.status_code == 200
    mock_vs.similarity_search_top_k.assert_awaited()
