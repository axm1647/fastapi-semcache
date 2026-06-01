"""FastAPI reverse proxy with semantic response caching."""

# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import Response

from .cache import SemanticCache
from .config import get_cache_settings
from .middleware.adapters.fastapi import SemanticCacheMiddleware

if TYPE_CHECKING:
    from aiohttp import ClientTimeout
    from fastapi import FastAPI

    from .config import CacheSettings

    _TimeoutArg = float | ClientTimeout
else:
    _TimeoutArg = float


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


def _require_aiohttp() -> ModuleType:
    """Import aiohttp or raise with install hint.

    Returns:
        The aiohttp module.

    Raises:
        ImportError: If aiohttp is not installed.
    """
    try:
        import aiohttp as _aiohttp
    except ImportError as exc:
        msg = (
            "Reverse proxy mode requires optional dependencies. "
            "pip install 'fastapi-semcache[proxy]'."
        )
        raise ImportError(msg) from exc
    return _aiohttp


def _require_fastapi() -> type[FastAPI]:
    """Import FastAPI or raise with install hint.

    Returns:
        The FastAPI application class.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI as _FastAPI
    except ImportError as exc:
        msg = (
            "Reverse proxy mode requires optional dependencies. "
            "pip install 'fastapi-semcache[proxy]'."
        )
        raise ImportError(msg) from exc
    return _FastAPI


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
        Header names and values suitable for the upstream HTTP client.
    """
    out: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower == "host":
            continue
        out[key] = value
    return out


def _filter_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
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
    timeout: _TimeoutArg = 300.0,
    verify: bool = True,
    aiohttp_session_kwargs: dict[str, object] | None = None,
    **middleware_kwargs: Any,
) -> FastAPI:
    """Build a FastAPI app that proxies to ``upstream`` behind ``SemanticCacheMiddleware``.

    The proxy forwards the request path and query string to the upstream base URL.
    Semantically cacheable requests (see middleware defaults) may return cached JSON
    without contacting upstream. Uncached successful JSON responses are stored.

    Streaming responses are buffered in full (same constraint as
    ``SemanticCacheMiddleware``).
    OpenAPI and interactive docs URLs honor ``disable_proxy_app_docs`` from
    ``get_cache_settings()`` at call time (not at import time).

    Requires the ``proxy`` optional extra (``fastapi`` and ``aiohttp``).

    Args:
        upstream: Base URL for the backend (for example ``http://127.0.0.1:8001`` or
            ``https://api.example.com/v1``). No trailing slash required.
        cache: Configured ``SemanticCache`` instance.
        timeout: Per-request timeout for upstream calls in seconds, or an ``aiohttp``
            ``ClientTimeout`` instance.
        verify: Whether to verify TLS certificates when ``upstream`` uses HTTPS.
        aiohttp_session_kwargs: Extra keyword arguments merged into
            ``aiohttp.ClientSession`` (for example a custom ``connector`` for tests
            or TLS settings).
        **middleware_kwargs: Forwarded to ``SemanticCacheMiddleware`` (``enabled``,
            ``path_prefix``, ``methods``, ``extract_query``, ``extract_model``,
            ``model_header_name``, ``validate_response``, ``cache_settings``,
            ``max_request_body_bytes``, ``max_response_body_bytes``).
            Optional 429 circuit breaker fields live on ``CacheSettings``
            (``SEMANTIC_CACHE_*`` env vars); see ``config.CacheSettings``.

    Returns:
        FastAPI application ready for ``uvicorn`` or another ASGI server.

    Raises:
        ValueError: If ``upstream`` is not a valid HTTP(S) URL with a host.
        ImportError: If the ``proxy`` extra (``fastapi`` and ``aiohttp``) is not
            installed.
    """
    FastAPI = _require_fastapi()
    aiohttp = _require_aiohttp()
    base = _validate_upstream(upstream)
    if isinstance(timeout, (int, float)):
        http_timeout = aiohttp.ClientTimeout(total=float(timeout))
    else:
        http_timeout = timeout

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Hold a shared ``aiohttp.ClientSession`` for the proxy lifetime.

        Args:
            app: FastAPI application.

        Yields:
            Control after startup and before shutdown.
        """
        extra = dict(aiohttp_session_kwargs or {})
        session_timeout = extra.pop("timeout", http_timeout)
        connector = extra.pop("connector", None)
        if connector is None:
            connector = aiohttp.TCPConnector(ssl=verify)
        async with aiohttp.ClientSession(
            timeout=session_timeout,
            connector=connector,
            **extra,
        ) as session:
            app.state.proxy_http_client = session
            app.state.proxy_upstream_base = base
            yield

    proxy_settings: CacheSettings = get_cache_settings()
    _disable_proxy_app_docs = proxy_settings.disable_proxy_app_docs
    _openapi_url = None if _disable_proxy_app_docs else "/openapi.json"
    _docs_url = None if _disable_proxy_app_docs else "/docs"
    _redoc_url = None if _disable_proxy_app_docs else "/redoc"

    app = FastAPI(
        lifespan=lifespan,
        title="Semantic cache proxy",
        openapi_url=_openapi_url,
        docs_url=_docs_url,
        redoc_url=_redoc_url,
    )
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
        client = request.app.state.proxy_http_client
        upstream_base: str = request.app.state.proxy_upstream_base
        path_component = f"/{full_path}" if full_path else "/"
        target = f"{upstream_base}{path_component}"
        if request.url.query:
            target = f"{target}?{request.url.query}"

        body = await request.body()
        req_headers = _forward_request_headers(request)

        try:
            async with client.request(
                request.method,
                target,
                data=body if body else None,
                headers=req_headers,
                allow_redirects=False,
            ) as upstream_resp:
                content = await upstream_resp.read()
                status = upstream_resp.status
                resp_headers = _filter_response_headers(upstream_resp.headers)
        except aiohttp.ClientError as exc:
            _logger.warning(
                "Upstream request failed (%s %s): %s",
                request.method,
                path_component,
                exc,
                exc_info=True,
            )
            return Response(
                content=b"Bad Gateway",
                status_code=502,
                media_type="text/plain; charset=utf-8",
            )

        return Response(
            content=content,
            status_code=status,
            headers=resp_headers,
        )

    return app
