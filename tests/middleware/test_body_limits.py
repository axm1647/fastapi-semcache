"""Tests for bounded request and response body buffering in ASGI middleware helpers."""

from __future__ import annotations

from typing import override
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.embedders import BaseEmbedder
from semanticcache.middleware.adapters.fastapi import SemanticCacheMiddleware
from semanticcache.middleware.adapters.fastapi.asgi_io import (
    call_downstream,
    read_body,
)


class _MiniEmbedder(BaseEmbedder):
    """Tiny deterministic embedder for middleware integration tests."""

    @property
    @override
    def embedding_dim(self) -> int:
        return 4

    @property
    @override
    def cache_namespace(self) -> str:
        return "middleware-body-limit-test"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vec = [0.25, 0.25, 0.25, 0.25]
        return [list(vec) for _ in texts]


def _mini_scope() -> Scope:
    """Minimal HTTP ASGI scope for unit tests."""
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "root_path": "",
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }


@pytest.mark.asyncio
async def test_read_body_accepts_within_limit() -> None:
    """Return full body when total length is at or below the cap."""
    queue: list[Message] = [
        {"type": "http.request", "body": b"12", "more_body": True},
        {"type": "http.request", "body": b"345", "more_body": False},
    ]

    async def receive() -> Message:
        return queue.pop(0)

    out = await read_body(receive, max_body_bytes=10)
    assert out == b"12345"


@pytest.mark.asyncio
async def test_read_body_raises_413_when_over_limit() -> None:
    """Raise HTTP 413 when the next chunk would exceed the cap."""
    queue: list[Message] = [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"X", "more_body": False},
    ]

    async def receive() -> Message:
        return queue.pop(0)

    with pytest.raises(HTTPException) as excinfo:
        await read_body(receive, max_body_bytes=4)
    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_read_body_drains_then_raises_on_overflow() -> None:
    """After overflow, remaining request chunks are consumed before raising."""
    drained: list[Message] = []

    queue: list[Message] = [
        {"type": "http.request", "body": b"12", "more_body": True},
        {"type": "http.request", "body": b"345", "more_body": True},
        {"type": "http.request", "body": b"678", "more_body": False},
    ]

    async def receive() -> Message:
        m = queue.pop(0)
        drained.append(m)
        return m

    with pytest.raises(HTTPException):
        await read_body(receive, max_body_bytes=3)
    assert len(drained) == 3


@pytest.mark.asyncio
async def test_call_downstream_returns_502_when_response_exceeds_limit() -> None:
    """Return HTTP 502 when buffered downstream body exceeds the cap."""

    async def huge_app(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {"type": "http.response.body", "body": b"a" * 100, "more_body": True}
        )
        await send(
            {"type": "http.response.body", "body": b"b" * 100, "more_body": False}
        )

    resp = await call_downstream(
        huge_app, _mini_scope(), b"", max_body_bytes=150
    )
    assert resp.status_code == 502
    assert b"Bad Gateway" in resp.body


@pytest.mark.asyncio
async def test_call_downstream_unlimited_when_max_is_none() -> None:
    """When max_body_bytes is None, the full downstream body is buffered."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {"type": "http.response.start", "status": 200, "headers": []}
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    resp = await call_downstream(app, _mini_scope(), b"", max_body_bytes=None)
    assert resp.status_code == 200
    assert resp.body == b"ok"


def test_middleware_request_body_limit_returns_413() -> None:
    """Integration: oversized POST body yields 413 before route runs."""
    cache = SemanticCache(
        embedder=_MiniEmbedder(),
        pg_uri="postgresql://mock/mock",
        redis_uri="",
        settings=CacheSettings(
            redis_uri=" ",
            pg_uri="postgresql://mock/mock",
            require_cache_scope=False,
        ),
    )

    app = FastAPI()

    @app.post("/echo")
    async def echo() -> JSONResponse:
        return JSONResponse({"ran": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cache,
        max_request_body_bytes=20,
        cache_settings=CacheSettings(require_cache_scope=False),
    )

    with TestClient(app) as client:
        r = client.post("/echo", content=b"x" * 50)

    assert r.status_code == 413


def test_middleware_response_body_limit_returns_502() -> None:
    """Integration: oversized downstream response yields 502 to the client."""
    cache = SemanticCache(
        embedder=_MiniEmbedder(),
        pg_uri="postgresql://mock/mock",
        redis_uri="",
        settings=CacheSettings(
            redis_uri=" ",
            pg_uri="postgresql://mock/mock",
            require_cache_scope=False,
        ),
    )
    mock_vs = AsyncMock()
    mock_vs.open = AsyncMock()
    mock_vs.ensure_schema = AsyncMock()
    mock_vs.similarity_search_top_k = AsyncMock(return_value=[])
    cache._vector_store = mock_vs

    app = FastAPI()

    @app.post("/big")
    async def big() -> JSONResponse:
        return JSONResponse({"payload": "z" * 500})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cache,
        max_response_body_bytes=80,
        cache_settings=CacheSettings(require_cache_scope=False),
    )

    with TestClient(app) as client:
        r = client.post("/big", json={"query": "hello"})

    assert r.status_code == 502


@pytest.mark.asyncio
async def test_read_body_none_means_unbounded() -> None:
    """When max_body_bytes is None, read_body buffers the full stream."""
    queue: list[Message] = [
        {"type": "http.request", "body": b"a" * 1000, "more_body": True},
        {"type": "http.request", "body": b"b" * 1000, "more_body": False},
    ]

    async def receive() -> Message:
        return queue.pop(0)

    out = await read_body(receive, max_body_bytes=None)
    assert len(out) == 2000
