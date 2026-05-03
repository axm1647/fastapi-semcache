"""FastAPI reverse proxy with semantic response caching."""

# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from starlette.responses import Response

from .cache import SemanticCache
from .middleware.fastapi import SemanticCacheMiddleware

_logger = logging.getLogger(__name__)

_HOP_BY_HOP: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

_PROXY_METHODS: tuple[str, ...] = (
    "GET",
    "HEAD",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
)


def _validate_upstream(url: str) -> str:
    """Normalize and validate an upstream base URL.

    Args:
        url: User-provided upstream (scheme, host, optional path prefix).

    Returns:
        Stripped URL without a trailing slash.

    Raises:
        ValueError: If the URL is not a usable HTTP(S) origin.
    """
    stripped = url.strip()
    if not stripped:
        msg = "upstream must be a non-empty URL"
        raise ValueError(msg)
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        msg = "upstream must use http or https"
        raise ValueError(msg)
    if not parsed.netloc:
        msg = "upstream must include a host"
        raise ValueError(msg)
    return stripped.rstrip("/")


def _forward_request_headers(request: Request) -> dict[str, str]:
    """Copy safe client headers for the upstream request.

    Args:
        request: Incoming ASGI request.

    Returns:
        Header names and values suitable for ``httpx``.
    """
    out: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower == "host":
            continue
        out[key] = value
    return out


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Drop hop-by-hop and length headers before returning to the client.

    Args:
        headers: Upstream response headers.

    Returns:
        Headers for the proxied ``Response`` (length set from body).
    """
    out: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower == "content-length":
            continue
        out[key] = value
    return out


def create_semantic_cache_proxy_app(
    *,
    upstream: str,
    cache: SemanticCache,
    timeout: float | httpx.Timeout = 300.0,
    verify: bool = True,
    httpx_client_kwargs: dict[str, Any] | None = None,
    **middleware_kwargs: Any,
) -> FastAPI:
    """Build a FastAPI app that proxies to ``upstream`` behind ``SemanticCacheMiddleware``.

    The proxy forwards the request path and query string to the upstream base URL.
    Semantically cacheable requests (see middleware defaults) may return cached JSON
    without contacting upstream. Uncached successful JSON responses are stored.

    Streaming responses are buffered in full (same constraint as
    ``SemanticCacheMiddleware``).

    Args:
        upstream: Base URL for the backend (for example ``http://127.0.0.1:8001`` or
            ``https://api.example.com/v1``). No trailing slash required.
        cache: Configured ``SemanticCache`` instance.
        timeout: Per-request timeout for upstream calls (seconds) or an ``httpx``
            timeout object.
        verify: Whether to verify TLS certificates when ``upstream`` uses HTTPS.
        httpx_client_kwargs: Extra keyword arguments merged into ``httpx.AsyncClient``
            (for example ``transport`` for tests or custom TLS settings).
        **middleware_kwargs: Forwarded to ``SemanticCacheMiddleware`` (``enabled``,
            ``path_prefix``, ``methods``, ``extract_query``, ``extract_model``,
            ``model_header_name``).

    Returns:
        FastAPI application ready for ``uvicorn`` or another ASGI server.

    Raises:
        ValueError: If ``upstream`` is not a valid HTTP(S) URL with a host.
    """
    base = _validate_upstream(upstream)
    http_timeout = (
        timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Hold a shared ``httpx.AsyncClient`` for the proxy lifetime.

        Args:
            app: FastAPI application.

        Yields:
            Control after startup and before shutdown.
        """
        extra = httpx_client_kwargs or {}
        merged: dict[str, Any] = {
            **extra,
            "follow_redirects": False,
            "timeout": http_timeout,
            "verify": verify,
        }
        async with httpx.AsyncClient(**merged) as client:
            app.state.proxy_http_client = client
            app.state.proxy_upstream_base = base
            yield

    app = FastAPI(lifespan=lifespan, title="Semantic cache proxy")
    app.add_middleware(SemanticCacheMiddleware, cache=cache, **middleware_kwargs)

    @app.api_route(
        "/{full_path:path}",
        methods=list(_PROXY_METHODS),
    )
    async def _proxy(request: Request, full_path: str) -> Response:
        """Forward the request to the configured upstream.

        Args:
            request: Incoming request (body may already be buffered by middleware).
            full_path: Path segment after the first slash.

        Returns:
            Proxied HTTP response.
        """
        client: httpx.AsyncClient = request.app.state.proxy_http_client
        upstream_base: str = request.app.state.proxy_upstream_base
        path_component = f"/{full_path}" if full_path else "/"
        target = f"{upstream_base}{path_component}"
        if request.url.query:
            target = f"{target}?{request.url.query}"

        body = await request.body()
        req_headers = _forward_request_headers(request)

        try:
            upstream_resp = await client.request(
                request.method,
                target,
                content=body if body else None,
                headers=req_headers,
            )
        except httpx.RequestError as exc:
            _logger.warning("Upstream request failed: %s", exc)
            return Response(
                content=f"Upstream error: {exc}".encode(),
                status_code=502,
                media_type="text/plain; charset=utf-8",
            )

        resp_headers = _filter_response_headers(upstream_resp.headers)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )

    return app
