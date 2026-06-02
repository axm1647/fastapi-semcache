"""Core type definitions for the semantic cache library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CacheSource = Literal[
    "embedders.sbert",
    "embedders.openai",
    "embedders.cohere",
    "embedders.voyage",
    "embedders.ollama",
    "none",
]


EmbedderType = Literal[
    "huggingface",
    "openai",
    "cohere",
    "voyage",
    "ollama",
]


@dataclass
class CacheQuery:
    """Request-shaped cache lookup payload (query text and optional model key)."""

    query: str
    model: str | None = None


@dataclass
class CacheResult:
    """Outcome of ``SemanticCache.get`` (hit or miss with optional payload)."""

    is_hit: bool
    similarity: float | None = None
    source: CacheSource = "none"
    response: dict[str, object] | None = None
    query_embedding: list[float] | None = field(default=None, repr=False)
    cache_entry_id: int | None = None


@dataclass
class CacheEntry:
    """One nearest-neighbor row from pgvector similarity search."""

    id: int
    query_text: str
    response: dict[str, object]
    similarity: float
