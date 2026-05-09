"""Tests for default and trusted scope extractors."""

from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from semanticcache.middleware.core.extractors import (
    default_extract_scope_from_request_context,
    trusted_extract_scope_from_server_side,
)


def _make_request(
    *,
    body: bytes = b"{}",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    """Build a Starlette request with optional raw body replay."""

    hdrs = headers or []

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "root_path": "",
        "query_string": b"",
        "headers": hdrs,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def receive() -> dict[str, str | bytes | bool]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


@pytest.mark.asyncio
async def test_default_extract_scope_reads_header_first() -> None:
    """Header scope wins over JSON when both are present."""

    body = json.dumps({"tenant_id": "from-json"}).encode()
    req = _make_request(
        body=body,
        headers=[(b"x-semantic-cache-scope", b"from-header")],
    )
    out = await default_extract_scope_from_request_context(
        req,
        body,
        scope_header_name="x-semantic-cache-scope",
    )
    assert out == "from-header"


@pytest.mark.asyncio
async def test_default_extract_scope_reads_json_tenant_id() -> None:
    """Numeric JSON tenant_id is normalized like body extraction."""

    body = json.dumps({"query": "hi", "tenant_id": 99}).encode()
    req = _make_request(body=body)
    out = await default_extract_scope_from_request_context(
        req,
        body,
        scope_header_name="x-semantic-cache-scope",
    )
    assert out == "99"


@pytest.mark.asyncio
async def test_trusted_extract_scope_ignores_client_headers_and_body() -> None:
    """Trusted helper does not read headers or body when state is unset."""

    body = json.dumps({"tenant_id": "spoofed"}).encode()
    req = _make_request(
        body=body,
        headers=[(b"x-semantic-cache-scope", b"spoofed-header")],
    )
    assert await trusted_extract_scope_from_server_side(req) is None


@pytest.mark.asyncio
async def test_trusted_extract_scope_reads_cache_scope_state() -> None:
    """request.state.cache_scope is returned when set."""

    req = _make_request()
    req.state.cache_scope = "org-1"
    assert await trusted_extract_scope_from_server_side(req) == "org-1"


@pytest.mark.asyncio
async def test_trusted_extract_scope_reads_tenant_id_state() -> None:
    """request.state.tenant_id integers stringify consistently."""

    req = _make_request()
    req.state.tenant_id = 42
    assert await trusted_extract_scope_from_server_side(req) == "42"


@pytest.mark.asyncio
async def test_trusted_extract_scope_prefers_cache_scope_over_tenant_id() -> None:
    """cache_scope is checked before tenant_id on state."""

    req = _make_request()
    req.state.cache_scope = "a"
    req.state.tenant_id = "b"
    assert await trusted_extract_scope_from_server_side(req) == "a"
