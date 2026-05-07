"""Compatibility exports for middleware replay helpers."""

from .core.replay import (
    build_hit_headers,
    build_miss_headers,
    cache_record_from_response,
    merge_response_headers,
    response_from_cache_hit,
)

__all__: list[str] = [
    "build_hit_headers",
    "build_miss_headers",
    "cache_record_from_response",
    "merge_response_headers",
    "response_from_cache_hit",
]
