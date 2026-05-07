"""Compatibility exports for middleware extraction helpers."""

from .core.extractors import (
    default_extract_model,
    default_extract_query,
    default_extract_scope,
)

__all__: list[str] = [
    "default_extract_model",
    "default_extract_query",
    "default_extract_scope",
]
