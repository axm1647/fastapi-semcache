"""Type stubs for the fastapi-semcache install-name compatibility module."""

from semanticcache import (
    DEFAULT_MAX_BODY_BYTES,
    CacheEntry,
    CacheQuery,
    CacheResult,
    ResponseShapeValidator,
    ResponseValidationContext,
    SemanticCache,
    SemanticCacheMiddleware,
    get_cache_settings,
    resolve_cache_scope,
)
from semanticcache.proxy import create_semantic_cache_proxy_app

__all__: list[str] = [
    "DEFAULT_MAX_BODY_BYTES",
    "CacheEntry",
    "CacheQuery",
    "CacheResult",
    "ResponseShapeValidator",
    "ResponseValidationContext",
    "SemanticCache",
    "SemanticCacheMiddleware",
    "create_semantic_cache_proxy_app",
    "get_cache_settings",
    "resolve_cache_scope",
]
