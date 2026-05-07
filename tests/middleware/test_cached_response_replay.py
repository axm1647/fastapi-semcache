"""Tests for replaying original response metadata on cache hits."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import cast

from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.middleware.fastapi import (
    ResponseValidationContext,
    SemanticCacheMiddleware,
)
from semanticcache.types import CacheResult


class _MemoryCache:
    """Store middleware cache payloads in memory for replay tests."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str | None, str], dict[str, object]] = {}

    @property
    def entry_count(self) -> int:
        """Return the number of stored cache entries."""
        return len(self._entries)

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
        key = (query, model, storage_scope_key or "")
        payload = self._entries.get(key)
        if payload is None:
            return CacheResult(is_hit=False)
        return CacheResult(
            is_hit=True,
            similarity=0.97,
            source="none",
            response=payload,
        )

    async def put(
        self,
        query: str,
        response: dict[str, object],
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> None:
        key = (query, model, storage_scope_key or "")
        self._entries[key] = response
        _ = scope


def test_cache_hit_replays_status_and_response_metadata() -> None:
    """Replay upstream status code and headers for cached JSON responses."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"count": 0}

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse(
            status_code=201,
            content={"ok": True},
            headers={"X-Upstream-Meta": "present"},
            media_type="application/problem+json",
        )

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
    )

    with TestClient(app) as client:
        first = client.post("/v1/chat", json={"query": "hi", "cache_scope": "tenant-a"})
        second = client.post(
            "/v1/chat",
            json={"query": "hi", "cache_scope": "tenant-a"},
        )

    assert first.status_code == 201
    assert first.headers.get("X-Cache") == "MISS"
    assert first.headers.get("X-Upstream-Meta") == "present"
    assert first.headers.get("content-type", "").startswith("application/problem+json")

    assert second.status_code == 201
    assert second.json() == {"ok": True}
    assert second.headers.get("X-Cache") == "HIT"
    assert second.headers.get("X-Upstream-Meta") == "present"
    assert second.headers.get("content-type", "").startswith("application/problem+json")
    assert calls["count"] == 1


def test_cache_store_skipped_when_cache_control_no_store() -> None:
    """Skip cache writes when upstream marks response as no-store."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"count": 0}

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse(
            status_code=200,
            content={"ok": True},
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
    )

    with TestClient(app) as client:
        first = client.post("/v1/chat", json={"query": "hi", "cache_scope": "tenant-a"})
        second = client.post(
            "/v1/chat",
            json={"query": "hi", "cache_scope": "tenant-a"},
        )

    assert first.status_code == 200
    assert first.headers.get("X-Cache") == "MISS"
    assert second.status_code == 200
    assert second.headers.get("X-Cache") == "MISS"
    assert calls["count"] == 2


def test_cache_store_skipped_when_cache_control_private() -> None:
    """Skip cache writes when upstream marks response as private."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"count": 0}

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse(
            status_code=200,
            content={"ok": True},
            headers={"Cache-Control": "private, max-age=120"},
        )

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
    )

    with TestClient(app) as client:
        first = client.post("/v1/chat", json={"query": "hi", "cache_scope": "tenant-a"})
        second = client.post(
            "/v1/chat",
            json={"query": "hi", "cache_scope": "tenant-a"},
        )

    assert first.status_code == 200
    assert first.headers.get("X-Cache") == "MISS"
    assert second.status_code == 200
    assert second.headers.get("X-Cache") == "MISS"
    assert calls["count"] == 2


def test_cache_store_skipped_when_set_cookie_present() -> None:
    """Skip cache writes when upstream includes Set-Cookie headers."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"count": 0}

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse(
            status_code=200,
            content={"ok": True},
            headers={"Set-Cookie": "session=abc123; HttpOnly; Path=/"},
        )

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
    )

    with TestClient(app) as client:
        first = client.post("/v1/chat", json={"query": "hi", "cache_scope": "tenant-a"})
        second = client.post(
            "/v1/chat",
            json={"query": "hi", "cache_scope": "tenant-a"},
        )

    assert first.status_code == 200
    assert first.headers.get("X-Cache") == "MISS"
    assert second.status_code == 200
    assert second.headers.get("X-Cache") == "MISS"
    assert calls["count"] == 2


def test_cache_store_skipped_when_response_validator_rejects_shape() -> None:
    """Skip cache writes when the response validator rejects the payload shape."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"count": 0}

    def validate_response(context: ResponseValidationContext) -> bool:
        """Accept only chat responses with OpenAI-style choices."""
        assert context.request.url.path == "/v1/chat"
        assert context.model == "gpt-5.4-mini"
        return isinstance(context.payload.get("choices"), list)

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse({"unexpected": True})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        validate_response=validate_response,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat",
            json={"query": "hi", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )
        second = client.post(
            "/v1/chat",
            json={"query": "hi", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )

    assert first.status_code == 200
    assert first.headers.get("X-Cache") == "MISS"
    assert second.status_code == 200
    assert second.headers.get("X-Cache") == "MISS"
    assert calls["count"] == 2
    assert cache.entry_count == 0


def test_response_validator_allows_route_and_model_matching_payload() -> None:
    """Store only payloads that match the expected route and model shape."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"count": 0}

    async def validate_response(context: ResponseValidationContext) -> bool:
        """Accept valid chat payloads for the expected provider model."""
        return (
            context.request.url.path == "/v1/chat"
            and context.model == "gpt-5.4-mini"
            and isinstance(context.payload.get("choices"), list)
        )

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse(
            {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
        validate_response=validate_response,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat",
            json={"query": "hi", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )
        second = client.post(
            "/v1/chat",
            json={"query": "hi", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )

    assert first.status_code == 200
    assert first.headers.get("X-Cache") == "MISS"
    assert second.status_code == 200
    assert second.headers.get("X-Cache") == "HIT"
    assert second.json() == {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}]
    }
    assert calls["count"] == 1


def test_cache_key_scopes_entries_by_request_path() -> None:
    """Avoid cross-endpoint cache hits for identical semantic request text."""
    app = FastAPI()
    cache = _MemoryCache()
    calls = {"chat": 0, "embeddings": 0}

    @app.post("/v1/chat")
    async def _chat_route() -> JSONResponse:
        calls["chat"] += 1
        return JSONResponse({"route": "chat"})

    @app.post("/v1/embeddings")
    async def _embeddings_route() -> JSONResponse:
        calls["embeddings"] += 1
        return JSONResponse({"route": "embeddings"})

    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cast(SemanticCache, cache),
    )

    with TestClient(app) as client:
        chat_first = client.post(
            "/v1/chat",
            json={"query": "same", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )
        embeddings_first = client.post(
            "/v1/embeddings",
            json={"query": "same", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )
        chat_second = client.post(
            "/v1/chat",
            json={"query": "same", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )
        embeddings_second = client.post(
            "/v1/embeddings",
            json={"query": "same", "model": "gpt-5.4-mini", "cache_scope": "tenant-a"},
        )

    assert chat_first.status_code == 200
    assert embeddings_first.status_code == 200
    assert chat_first.json() == {"route": "chat"}
    assert embeddings_first.json() == {"route": "embeddings"}
    assert chat_first.headers.get("X-Cache") == "MISS"
    assert embeddings_first.headers.get("X-Cache") == "MISS"
    assert chat_second.headers.get("X-Cache") == "HIT"
    assert embeddings_second.headers.get("X-Cache") == "HIT"
    assert chat_second.json() == {"route": "chat"}
    assert embeddings_second.json() == {"route": "embeddings"}
    assert calls["chat"] == 1
    assert calls["embeddings"] == 1
