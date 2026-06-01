"""Compatibility import for the fastapi-semcache distribution.

Preferred import:
    import semanticcache

Also supported:
    import fastapi_semcache
"""

from typing import TYPE_CHECKING

from semanticcache import CacheEntry as CacheEntry
from semanticcache import CacheQuery as CacheQuery
from semanticcache import CacheResult as CacheResult
from semanticcache import DEFAULT_MAX_BODY_BYTES as DEFAULT_MAX_BODY_BYTES
from semanticcache import ResponseShapeValidator as ResponseShapeValidator
from semanticcache import ResponseValidationContext as ResponseValidationContext
from semanticcache import SemanticCache as SemanticCache
from semanticcache import SemanticCacheMiddleware as SemanticCacheMiddleware
from semanticcache import get_cache_settings as get_cache_settings
from semanticcache import resolve_cache_scope as resolve_cache_scope

if TYPE_CHECKING:
    from semanticcache import create_semantic_cache_proxy_app

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
    """Lazily re-export the proxy helper without loading FastAPI at import time."""
    if name == "create_semantic_cache_proxy_app":
        from semanticcache import create_semantic_cache_proxy_app as proxy_factory

        return proxy_factory
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
