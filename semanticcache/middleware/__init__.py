"""HTTP middleware for semantic caching."""

from .fastapi import (
    SemanticCacheMiddleware,
    default_extract_query,
)

__all__: list[str] = ["SemanticCacheMiddleware", "default_extract_query"]
