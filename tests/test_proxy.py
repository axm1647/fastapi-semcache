"""Tests for ``create_semantic_cache_proxy_app`` routing and startup."""

from __future__ import annotations

from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from semanticcache.cache import SemanticCache
from semanticcache.proxy import create_semantic_cache_proxy_app
from semanticcache.types import CacheResult


class _MissCache:
    """Minimal cache that never hits (proxy forwards to upstream)."""

    async def get(self, query: str, model: str | None = None) -> CacheResult:
        _ = query, model
        return CacheResult(is_hit=False)

    async def put(
        self, query: str, response: dict[str, object], model: str | None = None
    ) -> None:
        _ = query, response, model


def test_create_proxy_rejects_invalid_upstream() -> None:
    """Invalid upstream URLs raise before the ASGI app is usable."""
    with pytest.raises(ValueError, match="upstream"):
        create_semantic_cache_proxy_app(
            upstream="",
            cache=cast(SemanticCache, _MissCache()),
            enabled=False,
        )


@pytest.mark.parametrize(
    ("method", "path", "expected_suffix"),
    [
        ("GET", "status", "http://upstream.local/status"),
        ("POST", "v1/chat", "http://upstream.local/v1/chat"),
    ],
)
def test_proxy_forwards_to_upstream(
    method: str,
    path: str,
    expected_suffix: str,
) -> None:
    """The catch-all route joins the configured upstream with the request path."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"proxied": True})

    transport = httpx.MockTransport(handler)
    app = create_semantic_cache_proxy_app(
        upstream="http://upstream.local",
        cache=cast(SemanticCache, _MissCache()),
        httpx_client_kwargs={"transport": transport},
        enabled=False,
    )
    url_path = f"/{path}" if not path.startswith("/") else path
    with TestClient(app) as client:
        r = client.request(method, url_path)
    assert r.status_code == 200
    assert r.json() == {"proxied": True}
    assert captured["url"] == expected_suffix


def test_proxy_app_includes_lifespan_client() -> None:
    """Startup wires ``proxy_http_client`` and ``proxy_upstream_base`` on state."""
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
