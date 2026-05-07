"""Tests for bypassing the cache when extractors raise."""

# pyright: reportPrivateUsage=false
# pyright: reportUnusedFunction=false
# pyright: reportCallIssue=false

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.middleware.fastapi import SemanticCacheMiddleware
from semanticcache.types import CacheResult


class _TrackingCache:
    """Records ``get`` / ``put`` calls for assertions."""

    def __init__(self) -> None:
        self.get_calls = 0
        self.put_calls = 0

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
        self.get_calls += 1
        _ = query, model, scope, storage_scope_key
        return CacheResult(is_hit=False)

    async def put(
        self,
        query: str,
        response: dict[str, object],
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> None:
        self.put_calls += 1
        _ = query, response, model, scope, storage_scope_key


@pytest.mark.asyncio
async def test_extract_query_exception_bypasses_cache_and_reaches_route() -> None:
    """Raising from ``extract_query`` skips cache IO and still invokes the app."""

    cache = _TrackingCache()

    async def extract_query(_request: Request, _body: bytes) -> str | None:
        raise RuntimeError("extract_query failed")

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"from_route": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        extract_query=extract_query,
    )

    with TestClient(app) as client:
        r = client.post("/v1/chat", json={"query": "hello"})

    assert r.status_code == 200
    assert r.json() == {"from_route": True}
    assert cache.get_calls == 0
    assert cache.put_calls == 0


@pytest.mark.asyncio
async def test_extract_model_exception_bypasses_cache_and_reaches_route() -> None:
    """Raising from ``extract_model`` skips cache IO and still invokes the app."""

    cache = _TrackingCache()

    async def extract_model(_request: Request, _body: bytes) -> str | None:
        raise RuntimeError("extract_model failed")

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"from_route": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        extract_model=extract_model,
    )

    with TestClient(app) as client:
        r = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert r.status_code == 200
    assert r.json() == {"from_route": True}
    assert cache.get_calls == 0
    assert cache.put_calls == 0


@pytest.mark.asyncio
async def test_extract_scope_exception_bypasses_cache_and_reaches_route() -> None:
    """Raising from ``extract_scope`` skips cache IO and still invokes the app."""

    cache = _TrackingCache()

    async def extract_scope(_request: Request, _body: bytes) -> str | None:
        raise RuntimeError("extract_scope failed")

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"from_route": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        extract_scope=extract_scope,
        cache_settings=CacheSettings(require_cache_scope=True),
    )

    with TestClient(app) as client:
        r = client.post("/v1/chat", json={"query": "hello"})

    assert r.status_code == 200
    assert r.json() == {"from_route": True}
    assert cache.get_calls == 0
    assert cache.put_calls == 0


@pytest.mark.asyncio
async def test_missing_scope_with_require_true_skips_cache_io() -> None:
    """Without a tenant scope, cache reads and writes are skipped (fail-closed)."""

    cache = _TrackingCache()

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        cache_settings=CacheSettings(require_cache_scope=True),
    )

    with TestClient(app) as client:
        r = client.post("/v1/chat", json={"query": "hello", "model": "m"})

    assert r.status_code == 200
    assert cache.get_calls == 0
    assert cache.put_calls == 0


@pytest.mark.asyncio
async def test_authorization_header_skips_cache_by_default() -> None:
    """Bypass cache IO when request includes ``Authorization`` by default."""
    cache = _TrackingCache()

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
    )

    with TestClient(app) as client:
        r = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer token"},
            json={"query": "hello", "cache_scope": "tenant-a"},
        )

    assert r.status_code == 200
    assert cache.get_calls == 0
    assert cache.put_calls == 0


@pytest.mark.asyncio
async def test_authorization_header_can_be_explicitly_cached() -> None:
    """Allow cache IO for ``Authorization`` requests when explicitly enabled."""
    cache = _TrackingCache()

    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        cache_settings=CacheSettings(cache_authorized_requests=True),
    )

    with TestClient(app) as client:
        r = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer token"},
            json={"query": "hello", "cache_scope": "tenant-a"},
        )

    assert r.status_code == 200
    assert cache.get_calls >= 1
    assert cache.put_calls == 1
