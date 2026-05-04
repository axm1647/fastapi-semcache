"""Tests for HTTP 429 consecutive circuit breaker in ``SemanticCacheMiddleware``."""

# pyright: reportCallIssue=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

import time
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.middleware.fastapi import SemanticCacheMiddleware
from semanticcache.types import CacheResult

_LLM_BODY: dict[str, list[dict[str, str]] | str] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "circuit breaker probe"}],
}


class _FakeSemanticCache:
    """Minimal async cache; set ``hit`` to True to force preflight hits."""

    def __init__(self, *, hit: bool = False) -> None:
        self.hit = hit

    async def get(self, query: str, model: str | None = None) -> CacheResult:
        _ = query
        if self.hit:
            return CacheResult(
                is_hit=True,
                similarity=0.99,
                source="embedders.sbert",
                response={"cached": True},
            )
        return CacheResult(
            is_hit=False,
            similarity=None,
            source="none",
            response=None,
        )

    async def put(
        self, query: str, response: dict[str, object], model: str | None = None
    ) -> None:
        _ = query, response, model


def _make_app(
    *,
    settings: CacheSettings,
    fake: _FakeSemanticCache,
    upstream_calls: list[int],
) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        upstream_calls[0] += 1
        return JSONResponse(status_code=429, content={"error": "rate_limited"})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, fake),
        cache_settings=settings,
    )
    return app


@pytest.fixture
def cb_settings() -> CacheSettings:
    """Circuit breaker enabled with a low trip threshold for tests."""
    return CacheSettings(
        circuit_breaker_429_enabled=True,
        circuit_breaker_429_consecutive_limit=2,
        circuit_breaker_429_open_seconds=60.0,
    )


def test_circuit_opens_after_consecutive_429_and_blocks_upstream(
    cb_settings: CacheSettings,
) -> None:
    """After N consecutive 429 responses, the next miss returns 503 without upstream."""
    calls = [0]
    fake = _FakeSemanticCache()
    app = _make_app(settings=cb_settings, fake=fake, upstream_calls=calls)
    client = TestClient(app)

    r1 = client.post("/v1/chat", json=_LLM_BODY)
    assert r1.status_code == 429
    assert calls[0] == 1

    r2 = client.post("/v1/chat", json=_LLM_BODY)
    assert r2.status_code == 429
    assert calls[0] == 2

    r3 = client.post("/v1/chat", json=_LLM_BODY)
    assert r3.status_code == 503
    assert r3.headers.get("X-Cache-Circuit") == "OPEN"
    assert calls[0] == 2


def test_non_429_resets_consecutive_counter(cb_settings: CacheSettings) -> None:
    """A successful upstream response clears the consecutive 429 counter."""
    calls = [0]
    fake = _FakeSemanticCache()
    app = FastAPI()

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls[0] += 1
        n = calls[0]
        if n == 1 or n >= 3:
            return JSONResponse(status_code=429, content={"error": "rate"})
        return JSONResponse(status_code=200, content={"ok": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, fake),
        cache_settings=cb_settings,
    )
    client = TestClient(app)

    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 429
    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 200
    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 429
    r4 = client.post("/v1/chat", json=_LLM_BODY)
    assert r4.status_code == 429
    r5 = client.post("/v1/chat", json=_LLM_BODY)
    assert r5.status_code == 503
    assert calls[0] == 4


def test_circuit_expires_and_upstream_resumes(cb_settings: CacheSettings) -> None:
    """After ``circuit_breaker_429_open_seconds``, upstream is contacted again."""
    short = cb_settings.model_copy(
        update={"circuit_breaker_429_open_seconds": 0.05},
    )
    calls = [0]
    fake = _FakeSemanticCache()
    app = _make_app(settings=short, fake=fake, upstream_calls=calls)
    client = TestClient(app)

    client.post("/v1/chat", json=_LLM_BODY)
    client.post("/v1/chat", json=_LLM_BODY)
    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 503
    assert calls[0] == 2

    time.sleep(0.08)
    r = client.post("/v1/chat", json=_LLM_BODY)
    assert r.status_code == 429
    assert calls[0] == 3


def test_cache_hit_bypasses_open_circuit(cb_settings: CacheSettings) -> None:
    """Preflight cache hits are returned even when the circuit is open."""
    calls = [0]
    fake = _FakeSemanticCache(hit=False)
    app = _make_app(settings=cb_settings, fake=fake, upstream_calls=calls)
    client = TestClient(app)

    client.post("/v1/chat", json=_LLM_BODY)
    client.post("/v1/chat", json=_LLM_BODY)
    assert client.post("/v1/chat", json=_LLM_BODY).status_code == 503

    fake.hit = True
    r = client.post("/v1/chat", json=_LLM_BODY)
    assert r.status_code == 200
    assert r.json() == {"cached": True}
    assert r.headers.get("X-Cache") == "HIT"
    assert calls[0] == 2
