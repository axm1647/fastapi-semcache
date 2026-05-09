"""Tests for ``get_embedder`` and ``CacheSettings`` embedder validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import semanticcache.embedders as embedders_mod
from semanticcache.config import CacheSettings
from semanticcache.embedders import get_embedder
from semanticcache.exceptions import NotSupportedEmbedderException


@pytest.mark.parametrize(
    "embedder_type",
    ("cohere", "voyage"),
)
def test_unsupported_types_raise(embedder_type: str) -> None:
    """Types not yet implemented must raise a clear error."""
    settings = CacheSettings.model_validate({"embedder_type": embedder_type})
    with pytest.raises(NotSupportedEmbedderException):
        get_embedder(settings)


@pytest.mark.parametrize(
    "embedder_type",
    ("not-a-real-type", "not-a-real-type-2"),
)
def test_invalid_embedder_type_raises_validation_error(embedder_type: str) -> None:
    """Reject embedder types outside the allowed literal union at settings parse time."""
    with pytest.raises(ValidationError):
        CacheSettings.model_validate({"embedder_type": embedder_type})


def test_huggingface_embedder_receives_settings_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass HF API key from settings into ``SBERTEmbedder`` construction."""
    captured: dict[str, str | None] = {}

    class _TrackingSBERT:
        """Capture init api key for factory wiring assertions."""

        def __init__(
            self, *args: object, api_key: str | None = None, **kwargs: object
        ) -> None:
            captured["api_key"] = api_key

    monkeypatch.setattr(embedders_mod, "SBERTEmbedder", _TrackingSBERT)
    settings = CacheSettings.model_validate(
        {
            "embedder_type": "huggingface",
            "SEMANTIC_CACHE_HUGGING_FACE_API_KEY": "hf-from-settings",
        }
    )

    _ = get_embedder(settings)
    assert captured["api_key"] == "hf-from-settings"


def test_ollama_requires_model_and_dimensions() -> None:
    """Ollama embedder type rejects settings without model id or dimension."""
    with pytest.raises(ValidationError):
        CacheSettings.model_validate({"embedder_type": "ollama"})
    with pytest.raises(ValidationError):
        CacheSettings.model_validate(
            {"embedder_type": "ollama", "ollama_embedding_model": "qwen3-embedding"}
        )


def test_ollama_embedder_constructed_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory passes URL, model, dimensions, and optional API key."""
    captured: dict[str, object] = {}

    class _TrackingOllama:
        """Capture constructor kwargs for factory wiring assertions."""

        def __init__(
            self,
            model_name: str,
            *,
            dimensions: int,
            api_key: str | None = None,
            base_url: str = "",
        ) -> None:
            captured["model_name"] = model_name
            captured["dimensions"] = dimensions
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(embedders_mod, "OllamaEmbedder", _TrackingOllama)
    settings = CacheSettings.model_validate(
        {
            "embedder_type": "ollama",
            "ollama_embedding_model": "my-embed-model",
            "ollama_embedding_dimensions": 1024,
            "ollama_base_url": "http://embeddings.example:11434/v1",
            "SEMANTIC_CACHE_OLLAMA_API_KEY": "k",
        }
    )
    _ = get_embedder(settings)
    assert captured["model_name"] == "my-embed-model"
    assert captured["dimensions"] == 1024
    assert captured["api_key"] == "k"
    assert captured["base_url"] == "http://embeddings.example:11434/v1"
