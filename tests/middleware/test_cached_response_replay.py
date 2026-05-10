"""Tests for replaying original response metadata on cache hits."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import cast

from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.middleware.adapters.fastapi import (
    ResponseValidationContext,
    SemanticCacheMiddleware,
)
from semanticcache.middleware.adapters.fastapi.cache_ops import response_allows_cache_store
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


class _UnwrappedPayloadEvictionCache:
    """Serve a similarity hit without replay envelope, then miss until ``put`` stores wrapped rows."""

    MARKER = "__semanticcache_record_v1__"

    def __init__(self) -> None:
        """Initialize eviction bookkeeping for plain-body hits."""
        self.entry_id = 200
        self.deleted_ids: list[int] = []
        self._plain_hit_cleared = False
        self._wrapped: dict[str, object] | None = None

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
        _ = query, model, scope, storage_scope_key
        if self._wrapped is not None:
            return CacheResult(
                is_hit=True,
                similarity=0.97,
                source="none",
                cache_entry_id=self.entry_id,
                response=self._wrapped,
            )
        if self._plain_hit_cleared:
            return CacheResult(
                is_hit=False,
                query_embedding=[0.25, 0.25, 0.25, 0.25],
            )
        return CacheResult(
            is_hit=True,
            similarity=0.95,
            source="none",
            cache_entry_id=self.entry_id,
            response={"plain_body": True},
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
        self._wrapped = response

    async def delete_entry_by_id(
        self,
        entry_id: int,
        *,
        model: str | None = None,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> bool:
        """Mark plain-body hit removed after eviction."""
        _ = model, scope, storage_scope_key
        self.deleted_ids.append(entry_id)
        self._plain_hit_cleared = True
        return True


class _CorruptHitEvictionCache:
    """Return a miss once, then an unreplayable hit until eviction clears storage."""

    MARKER = "__semanticcache_record_v1__"

    def __init__(self) -> None:
        """Initialize empty storage and eviction bookkeeping."""
        self._stored: dict[str, object] | None = None
        self.entry_id = 100
        self.deleted_ids: list[int] = []

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
        _ = query, model, scope
        if self._stored is None:
            return CacheResult(
                is_hit=False,
                query_embedding=[0.25, 0.25, 0.25, 0.25],
            )
        return CacheResult(
            is_hit=True,
            similarity=0.95,
            source="none",
            cache_entry_id=self.entry_id,
            response=cast(
                dict[str, object],
                {
                    self.MARKER: True,
                    "body": ["not", "a", "dict"],
                },
            ),
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
        self._stored = response

    async def delete_entry_by_id(
        self,
        entry_id: int,
        *,
        model: str | None = None,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> bool:
        """Record eviction and clear stored payload for subsequent misses."""
        _ = model, scope, storage_scope_key
        self.deleted_ids.append(entry_id)
        self._stored = None
        self.entry_id += 1
        return True


def test_plain_json_hit_without_envelope_is_evicted_and_rewrapped_on_miss() -> None:
    """Similarity hits without replay envelope are not served; eviction allows a wrapped store."""
    app = FastAPI()
    cache = _UnwrappedPayloadEvictionCache()
    calls = {"count": 0}

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse(status_code=201, content={"created": True})

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
    assert second.status_code == 201
    assert second.headers.get("X-Cache") == "HIT"
    assert second.json() == {"created": True}
    assert calls["count"] == 1
    assert cache.deleted_ids == [200]


def test_unreplayable_hit_is_logged_evicted_and_calls_upstream() -> None:
    """Corrupt replay-shaped rows are removed and the handler runs as on miss."""
    app = FastAPI()
    cache = _CorruptHitEvictionCache()
    calls = {"count": 0}

    @app.post("/v1/chat")
    async def _route() -> JSONResponse:
        calls["count"] += 1
        return JSONResponse({"ok": True})

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
    assert cache.deleted_ids == [100]


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


def test_cache_store_skipped_when_raw_set_cookie_present() -> None:
    """Skip cache writes when raw headers include Set-Cookie."""

    class _Headers:
        """Expose a mapping API that omits Set-Cookie."""

        def get(self, name: str, default: str | None = None) -> str | None:
            """Return default for all headers."""
            _ = name
            return default

    class _Response:
        """Expose raw Set-Cookie headers with a sparse header mapping."""

        headers = _Headers()
        raw_headers = [
            (b"content-type", b"application/json"),
            (b"set-cookie", b"session=abc123; HttpOnly; Path=/"),
        ]

    assert not response_allows_cache_store(cast(JSONResponse, _Response()))


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
