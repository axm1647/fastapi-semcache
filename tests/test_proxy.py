"""Tests for ``create_semantic_cache_proxy_app`` routing and startup."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock
from urllib.parse import urlparse

import aiohttp
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.proxy import create_semantic_cache_proxy_app
from semanticcache.types import CacheResult


class _MissCache:
    """Minimal cache that never hits (proxy forwards to upstream)."""

    async def get(
        self,
        query: str,
        model: str | None = None,
        *,
        scope: str | None = None,
        storage_scope_key: str | None = None,
    ) -> CacheResult:
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
        _ = query, response, model, scope, storage_scope_key


class _FakeUpstreamResponse:
    """Minimal aiohttp response stand-in for proxy forwarding tests."""

    status = 200
    headers: dict[str, str] = {}

    async def read(self) -> bytes:
        return b'{"proxied": true}'

    async def __aenter__(self) -> _FakeUpstreamResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeClientSession:
    """Record upstream requests and return a canned JSON response."""

    def __init__(self, captured: dict[str, str]) -> None:
        self._captured = captured

    async def __aenter__(self) -> _FakeClientSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: object) -> _FakeUpstreamResponse:
        _ = method, kwargs
        self._captured["url"] = url
        return _FakeUpstreamResponse()


def test_create_proxy_rejects_invalid_upstream() -> None:
    """Invalid upstream URLs raise before the ASGI app is usable."""
    with pytest.raises(ValueError, match="upstream"):
        create_semantic_cache_proxy_app(
            upstream="",
            cache=cast(SemanticCache, _MissCache()),
            enabled=False,
        )


@pytest.mark.parametrize(
    ("method", "path", "expected_path"),
    [
        ("GET", "status", "/status"),
        ("POST", "v1/chat", "/v1/chat"),
    ],
)
def test_proxy_forwards_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    expected_path: str,
) -> None:
    """The catch-all route joins the configured upstream with the request path."""
    captured: dict[str, str] = {}

    def fake_client_session(**kwargs: object) -> _FakeClientSession:
        _ = kwargs
        return _FakeClientSession(captured)

    monkeypatch.setattr(aiohttp, "ClientSession", fake_client_session)
    monkeypatch.setattr(aiohttp, "TCPConnector", lambda **kwargs: AsyncMock())

    app = create_semantic_cache_proxy_app(
        upstream="http://upstream.local",
        cache=cast(SemanticCache, _MissCache()),
        enabled=False,
    )
    url_path = f"/{path}" if not path.startswith("/") else path
    with TestClient(app) as client:
        r = client.request(method, url_path)
    assert r.status_code == 200
    assert r.json() == {"proxied": True}
    assert urlparse(captured["url"]).path == expected_path


def test_proxy_app_includes_lifespan_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup wires ``proxy_http_client`` and ``proxy_upstream_base`` on state."""
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: _FakeClientSession({}))
    monkeypatch.setattr(aiohttp, "TCPConnector", lambda **kwargs: AsyncMock())

    app = create_semantic_cache_proxy_app(
        upstream="http://127.0.0.1:1",
        cache=cast(SemanticCache, _MissCache()),
        enabled=False,
    )
    with TestClient(app) as client:
        fa = cast(FastAPI, client.app)
        state = fa.state
        assert hasattr(state, "proxy_http_client")
        assert hasattr(state, "proxy_upstream_base")
        assert state.proxy_upstream_base == "http://127.0.0.1:1"
