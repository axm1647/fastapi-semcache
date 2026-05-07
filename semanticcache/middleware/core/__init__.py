"""Framework-agnostic middleware core helpers."""

from .coordination import MiddlewareCoordination
from .extractors import (
    default_extract_model,
    default_extract_query,
    default_extract_scope,
)
from .replay import (
    build_hit_headers,
    build_miss_headers,
    cache_record_from_response,
    merge_response_headers,
    response_from_cache_hit,
)

__all__: list[str] = [
    "MiddlewareCoordination",
    "build_hit_headers",
    "build_miss_headers",
    "cache_record_from_response",
    "default_extract_model",
    "default_extract_query",
    "default_extract_scope",
    "merge_response_headers",
    "response_from_cache_hit",
]
