"""Type stubs for optional reverse-proxy helpers."""

from collections.abc import Awaitable, Callable, Sequence
from typing import TypedDict

from aiohttp import ClientTimeout
from fastapi import FastAPI
from starlette.requests import Request

from .cache import SemanticCache
from .config import CacheSettings
from .middleware import ResponseShapeValidator

class SemanticCacheProxyMiddlewareKwargs(TypedDict, total=False):
    """Keyword arguments forwarded to ``SemanticCacheMiddleware``."""

    enabled: bool
    path_prefix: str | None
    methods: Sequence[str]
    extract_query: Callable[[Request, bytes], Awaitable[str | None]]
    extract_model: Callable[[Request, bytes], Awaitable[str | None]] | None
    model_header_name: str
    extract_scope: Callable[[Request, bytes], Awaitable[str | None]] | None
    scope_header_name: str
    validate_response: ResponseShapeValidator | None
    cache_settings: CacheSettings | None
    max_request_body_bytes: int | None
    max_response_body_bytes: int | None


def create_semantic_cache_proxy_app(
    *,
    upstream: str,
    cache: SemanticCache,
    timeout: float | ClientTimeout = 300.0,
    verify: bool = True,
    aiohttp_session_kwargs: dict[str, object] | None = None,
    **middleware_kwargs: SemanticCacheProxyMiddlewareKwargs,
) -> FastAPI: ...
