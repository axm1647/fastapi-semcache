"""Shared pytest configuration for the fastapi-semcache test suite."""

from __future__ import annotations

import os

import pytest

_SEMANTIC_CACHE_PREFIX = "SEMANTIC_CACHE_"
# Non-prefixed env names that ``CacheSettings`` reads via ``validation_alias``.
_CACHE_SETTINGS_ALIAS_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "COHERE_API_KEY",
        "VOYAGE_API_KEY",
        "OLLAMA_API_KEY",
        "HUGGINGFACE_API_KEY",
    }
)


@pytest.fixture(autouse=True)
def _isolate_cache_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip cache-related env vars so ``CacheSettings()`` is deterministic.

    Developers often export ``SEMANTIC_CACHE_*`` from ``.env`` for local apps.
    Because ``CacheSettings`` is a ``BaseSettings`` subclass, those variables
    would otherwise override constructor kwargs and break unit tests.
    """
    for key in list(os.environ):
        if key.startswith(_SEMANTIC_CACHE_PREFIX) or key in _CACHE_SETTINGS_ALIAS_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
