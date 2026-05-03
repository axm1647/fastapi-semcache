"""Semantic caching: embeddings, pgvector, optional Redis."""

from semanticcache.cache import SemanticCache
from semanticcache.config import get_cache_settings
from semanticcache.middleware import SemanticCacheMiddleware
from semanticcache.types import CacheEntry, CacheQuery, CacheResult

__all__: list[str] = [
    "CacheEntry",
    "CacheQuery",
    "CacheResult",
    "SemanticCache",
    "SemanticCacheMiddleware",
    "get_cache_settings",
]
