"""Framework adapter entrypoints for semantic cache middleware."""

from .fastapi import SemanticCacheMiddleware

__all__: list[str] = ["SemanticCacheMiddleware"]
