"""End-to-end streaming and tee-mode middleware tests."""

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.middleware.adapters.fastapi import SemanticCacheMiddleware
from semanticcache.types import CacheResult

_MARKER = "__semanticcache_record_v1__"

_LLM_BODY: dict[str, object] = {
    "model": "stream-model",
    "cache_scope": "tee-tenant",
    "messages": [{"role": "user", "content": "streamed once"}],
}


def _hit_record(body: dict[str, object]) -> dict[str, object]:
    """Build a replayable cache row for a synthetic hit."""
    return {
        _MARKER: True,
        "body": body,
        "meta": {
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "media_type": "application/json",
        },
    }


class _StreamingFakeCache:
    """Records ``put`` and serves one miss then a hit."""

    def __init__(self) -> None:
        self.put_count = 0
        self.last_record: dict[str, object] | None = None
        self._serve_hit = False

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
        _ = query, scope, storage_scope_key
        if self._serve_hit and self.last_record is not None:
            body_obj = self.last_record.get("body")
            body = body_obj if isinstance(body_obj, dict) else {}
            return CacheResult(
                is_hit=True,
                similarity=0.99,
                source="embedders.sbert",
                response=_hit_record(body),
            )
        return CacheResult(
            is_hit=False,
            similarity=None,
            source="none",
            response=None,
            query_embedding=[0.1, 0.2, 0.3, 0.4],
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
        _ = query, model, scope, storage_scope_key, query_embedding
        self.put_count += 1
        self.last_record = response
        self._serve_hit = True


def test_tee_mode_e2e_miss_then_hit_json_response() -> None:
    """Tee path completes, stores once, second request is a cache hit.

    Chunk-level streaming is covered by ``TeeSend`` / ``stream_tee_and_store``
    tests. ``TestClient`` can hang when the inner route returns
    ``StreamingResponse`` under this middleware (the same occurs in buffered
    mode), so this integration uses a normal ``JSONResponse``.
    """
    fake = _StreamingFakeCache()
    app = FastAPI()

    @app.post("/v1/chat")
    async def _chat() -> JSONResponse:
        return JSONResponse({"streamed": True})

    settings = CacheSettings(
        response_mode="tee",
        require_cache_scope=True,
        circuit_breaker_429_enabled=False,
    )
    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, fake),
        cache_settings=settings,
    )

    client = TestClient(app)
    r1 = client.post("/v1/chat", json=_LLM_BODY)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"
    assert r1.json() == {"streamed": True}

    # A second request re-enters the anyio event loop, which drives the
    # background _store() task to completion before the response returns.
    # time.sleep would block the OS thread and starve the event loop.
    r2 = client.post("/v1/chat", json=_LLM_BODY)

    assert fake.put_count == 1
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") == "HIT"
    assert r2.json() == {"streamed": True}


def test_tee_mode_defaults_hit_response_mode_to_stream() -> None:
    """response_mode=tee automatically sets hit_response_mode=stream when unset."""
    settings = CacheSettings(response_mode="tee")
    assert settings.hit_response_mode == "stream"


def test_buffered_mode_leaves_hit_response_mode_as_single() -> None:
    """response_mode=buffered does not change the hit_response_mode default."""
    settings = CacheSettings(response_mode="buffered")
    assert settings.hit_response_mode == "single"


def test_tee_mode_explicit_single_hit_response_mode_is_respected() -> None:
    """Explicit hit_response_mode=single overrides the tee auto-coupling."""
    settings = CacheSettings(response_mode="tee", hit_response_mode="single")
    assert settings.hit_response_mode == "single"


def test_hit_stream_mode_emits_asgi_chunks_without_content_length() -> None:
    """tee mode auto-applies stream hit delivery; hits have no content-length.

    hit_response_mode is not set explicitly -- the coupled default is exercised.
    TestClient reassembles chunked ASGI body messages transparently, so body
    equality and status code are the primary observable assertions here.
    """
    fake = _StreamingFakeCache()
    app = FastAPI()

    @app.post("/v1/chat")
    async def _chat() -> JSONResponse:
        return JSONResponse({"streamed": True})

    settings = CacheSettings(
        response_mode="tee",
        require_cache_scope=True,
        circuit_breaker_429_enabled=False,
    )
    assert settings.hit_response_mode == "stream"
    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, fake),
        cache_settings=settings,
    )

    client = TestClient(app)
    r1 = client.post("/v1/chat", json=_LLM_BODY)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"
    assert r1.json() == {"streamed": True}

    r2 = client.post("/v1/chat", json=_LLM_BODY)
    assert fake.put_count == 1
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") == "HIT"
    assert r2.json() == {"streamed": True}
    assert "content-length" not in {k.lower() for k in r2.headers}


def test_hit_stream_mode_chunk_size_splits_body() -> None:
    """hit_stream_chunk_size controls how many ASGI body messages are emitted.

    We intercept the raw ASGI send callable by patching the middleware's ASGI
    ``__call__`` path via a thin wrapper app so we can count body chunks.
    """
    from starlette.types import Message, Receive, Scope, Send

    fake = _StreamingFakeCache()
    inner_app = FastAPI()

    @inner_app.post("/v1/chat")
    async def _chat() -> JSONResponse:
        return JSONResponse({"streamed": True})

    settings = CacheSettings(
        response_mode="tee",
        hit_response_mode="stream",
        hit_stream_chunk_size=2,
        require_cache_scope=True,
        circuit_breaker_429_enabled=False,
    )
    inner_app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, fake),
        cache_settings=settings,
    )

    body_messages: list[Message] = []

    async def intercepting_app(scope: Scope, receive: Receive, send: Send) -> None:
        async def recording_send(msg: Message) -> None:
            if msg["type"] == "http.response.body":
                body_messages.append(msg)
            await send(msg)

        await inner_app(scope, receive, recording_send)

    from starlette.applications import Starlette
    from starlette.testclient import TestClient as _TC

    outer = Starlette()
    outer.mount("/", intercepting_app)  # type: ignore[arg-type]
    client = _TC(outer)

    client.post("/v1/chat", json=_LLM_BODY)
    body_messages.clear()
    r2 = client.post("/v1/chat", json=_LLM_BODY)

    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") == "HIT"
    assert len(body_messages) > 1
    for msg in body_messages[:-1]:
        assert msg.get("more_body") is True
    assert body_messages[-1].get("more_body") is False


def test_tee_mode_circuit_breaker_counts_429() -> None:
    """429 responses in tee mode still advance the consecutive 429 counter."""

    class _MissCache:
        async def get(
            self,
            query: str,
            model: str | None = None,
            *,
            scope: str | None = None,
            storage_scope_key: str | None = None,
        ) -> CacheResult:
            _ = query, model, scope, storage_scope_key
            return CacheResult(is_hit=False, similarity=None, source="none")

        async def put(
            self,
            query: str,
            response: dict[str, object],
            model: str | None = None,
            *,
            scope: str | None = None,
            storage_scope_key: str | None = None,
        ) -> None:
            _ = query, response, model, scope, storage_scope_key

    calls = [0]
    app = FastAPI()

    @app.post("/v1/chat")
    async def _rate_limited() -> JSONResponse:
        calls[0] += 1
        return JSONResponse(status_code=429, content={"error": "rate"})

    settings = CacheSettings(
        response_mode="tee",
        require_cache_scope=True,
        circuit_breaker_429_enabled=True,
        circuit_breaker_429_consecutive_limit=2,
        circuit_breaker_429_open_seconds=60.0,
    )
    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, _MissCache()),
        cache_settings=settings,
    )

    client = TestClient(app)
    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 429
    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 429
    r3 = client.post("/v1/chat", json=_LLM_BODY)
    assert r3.status_code == 503
    assert r3.headers.get("X-Cache-Circuit") == "OPEN"
    assert calls[0] == 2
