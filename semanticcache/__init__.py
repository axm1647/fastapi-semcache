"""Semantic caching: embeddings, pgvector, optional Redis."""

from semanticcache.cache import SemanticCache
from semanticcache.config import get_cache_settings
from semanticcache.middleware import SemanticCacheMiddleware
from semanticcache.proxy import create_semantic_cache_proxy_app
from semanticcache.types import CacheEntry, CacheQuery, CacheResult

__all__: list[str] = [
    "CacheEntry",
    "CacheQuery",
    "CacheResult",
    "SemanticCache",
    "SemanticCacheMiddleware",
    "create_semantic_cache_proxy_app",
    "get_cache_settings",
]
