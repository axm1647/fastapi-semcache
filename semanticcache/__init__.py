"""Semantic caching: embeddings, pgvector, optional Redis."""

from typing import TYPE_CHECKING

from .cache import SemanticCache, resolve_cache_scope
from .config import get_cache_settings
from .middleware import (
    DEFAULT_MAX_BODY_BYTES,
    ResponseShapeValidator,
    ResponseValidationContext,
    SemanticCacheMiddleware,
)
from .types import CacheEntry, CacheQuery, CacheResult

if TYPE_CHECKING:
    from .proxy import create_semantic_cache_proxy_app

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


def __getattr__(name: str) -> object:
    """Lazily expose optional proxy helpers without importing FastAPI at load time."""
    if name == "create_semantic_cache_proxy_app":
        from .proxy import create_semantic_cache_proxy_app as proxy_factory

        return proxy_factory
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
