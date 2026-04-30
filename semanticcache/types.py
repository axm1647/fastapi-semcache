# pyright: reportAny=false
# pyright: reportExplicitAny=false

from typing import Any, Literal

from pydantic import BaseModel


class CacheQuery(BaseModel):
    query: str
    model: str | None = None


class CacheResult(BaseModel):
    is_hit: bool
    similarity: float | None = None
    source: Literal["embedders.sbert", "embedders.openai", "none"] = "none"
    response: dict[str, Any] | None = None
