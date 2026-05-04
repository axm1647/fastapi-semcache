from typing import Literal

from pydantic import BaseModel


class CacheQuery(BaseModel):
    """Request-shaped cache lookup payload (query text and optional model key)."""

    query: str
    model: str | None = None


class CacheResult(BaseModel):
    """Outcome of ``SemanticCache.get`` (hit or miss with optional payload)."""

    is_hit: bool
    similarity: float | None = None
    source: Literal[
        "embedders.sbert",
        "embedders.openai",
        "embedders.cohere",
        "embedders.voyage",
        "embedders.ollama",
        "none",
    ] = "none"
    response: dict[str, object] | None = None


class CacheEntry(BaseModel):
    """One nearest-neighbor row from pgvector similarity search."""

    id: int
    query_text: str
    response: dict[str, object]
    similarity: float
