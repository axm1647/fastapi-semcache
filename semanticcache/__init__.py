"""Semantic caching: embeddings, pgvector, optional Redis."""

from .cache import SemanticCache, resolve_cache_scope
from .config import get_cache_settings
from .middleware import (
    DEFAULT_MAX_BODY_BYTES,
    ResponseShapeValidator,
    ResponseValidationContext,
    SemanticCacheMiddleware,
)
from .proxy import create_semantic_cache_proxy_app
from .types import CacheEntry, CacheQuery, CacheResult

__all__: list[str] = [
    "CacheEntry",
    "CacheQuery",
    "CacheResult",
    "DEFAULT_MAX_BODY_BYTES",
    "ResponseShapeValidator",
    "ResponseValidationContext",
    "resolve_cache_scope",
    "SemanticCache",
    "SemanticCacheMiddleware",
    "create_semantic_cache_proxy_app",
    "get_cache_settings",
]
