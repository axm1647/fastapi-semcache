"""Tests for bypassing the cache when extractors raise."""

# pyright: reportPrivateUsage=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.middleware.fastapi import SemanticCacheMiddleware
from semanticcache.types import CacheResult


class _TrackingCache:
    """Records ``get`` / ``put`` calls for assertions."""

    def __init__(self) -> None:
        self.get_calls = 0
        self.put_calls = 0

    async def get(self, query: str, model: str | None = None) -> CacheResult:
        self.get_calls += 1
        _ = query, model
        return CacheResult(is_hit=False)

    async def put(
        self, query: str, response: dict[str, object], model: str | None = None
    ) -> None:
        self.put_calls += 1
        _ = query, response, model


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
