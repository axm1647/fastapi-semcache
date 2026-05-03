"""HTTP middleware for semantic caching."""

from semanticcache.middleware.fastapi import (
    SemanticCacheMiddleware,
    default_extract_query,
)

__all__: list[str] = ["SemanticCacheMiddleware", "default_extract_query"]
