"""Integration tests for ``stream_tee_and_store``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from semanticcache.middleware.adapters.fastapi.flow import LookupContext, stream_tee_and_store


def _scope() -> Scope:
    """Build a minimal HTTP ASGI scope."""
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/test",
        "raw_path": b"/v1/test",
        "root_path": "",
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }


@pytest.mark.asyncio
async def test_stream_tee_and_store_forwards_chunks_and_calls_cache_put() -> None:
    """Client receives all chunks in order; ``cache_put`` runs once after stream."""
    client_bodies: list[bytes] = []

    async def capture_send(message: Message) -> None:
        if message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if isinstance(chunk, bytes) and chunk:
                client_bodies.append(chunk)

    async def tri_chunk_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            },
        )
        await send(
            {"type": "http.response.body", "body": b'{"a"', "more_body": True},
        )
        await send(
            {"type": "http.response.body", "body": b": 1", "more_body": True},
        )
        await send(
            {"type": "http.response.body", "body": b"}", "more_body": False},
        )

    cache_put = AsyncMock()
    scope = _scope()
    request = Request(scope)
    lookup = LookupContext(
        query="q",
        model="m",
        raw_scope="tenant",
        scope_storage="tenant",
    )

    async def shape_ok(
        req: Request,
        req_body: bytes,
        resp: object,
        payload: dict[str, object],
        model: str | None,
        raw_scope: str | None,
    ) -> bool:
        return True

    await stream_tee_and_store(
        app=tri_chunk_app,
        scope=scope,
        body=b"{}",
        send=capture_send,
        lookup_ctx=lookup,
        request=request,
        query_embedding=[0.1, 0.2],
        max_body_bytes=None,
        miss_headers={"X-Cache": "MISS"},
        response_allows_cache_store=lambda r: True,
        response_shape_allows_cache_store=shape_ok,
        cache_record_from_response=lambda p, r: p,
        cache_put=cache_put,
    )

    await asyncio.sleep(0.05)

    assert client_bodies == [b'{"a"', b": 1", b"}"]
    cache_put.assert_awaited_once()
    call_kw = cache_put.await_args
    assert call_kw is not None
    args, kwargs = call_kw
    assert args[0] == "q"
    assert args[1] == {"a": 1}
    assert args[2] == "m"
    assert args[3] == "tenant"
    assert args[4] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_stream_tee_and_store_over_limit_skips_cache_put() -> None:
    """Oversized tee buffer does not call ``cache_put``; client still gets bytes."""
    client_bodies: list[bytes] = []

    async def capture_send(message: Message) -> None:
        if message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if isinstance(chunk, bytes):
                client_bodies.append(chunk)

    async def huge_chunk_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            },
        )
        await send(
            {"type": "http.response.body", "body": b"123456", "more_body": False},
        )

    cache_put = AsyncMock()
    scope = _scope()
    request = Request(scope)
    lookup = LookupContext(
        query="q",
        model=None,
        raw_scope="t",
        scope_storage="t",
    )

    async def shape_ok(
        req: Request,
        req_body: bytes,
        resp: object,
        payload: dict[str, object],
        model: str | None,
        raw_scope: str | None,
    ) -> bool:
        return True

    await stream_tee_and_store(
        app=huge_chunk_app,
        scope=scope,
        body=b"{}",
        send=capture_send,
        lookup_ctx=lookup,
        request=request,
        query_embedding=None,
        max_body_bytes=5,
        miss_headers={},
        response_allows_cache_store=lambda r: True,
        response_shape_allows_cache_store=shape_ok,
        cache_record_from_response=lambda p, r: p,
        cache_put=cache_put,
    )

    await asyncio.sleep(0.05)

    assert client_bodies == [b"123456"]
    cache_put.assert_not_awaited()
