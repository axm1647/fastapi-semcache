"""Semantic caching: embeddings, pgvector, optional Redis."""

from semanticcache.cache import SemanticCache
from semanticcache.config import get_cache_settings
from semanticcache.types import CacheEntry, CacheQuery, CacheResult

__all__: list[str] = [
    "CacheEntry",
    "CacheQuery",
    "CacheResult",
    "SemanticCache",
    "get_cache_settings",
]
