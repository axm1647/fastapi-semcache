"""Tests for ``get_embedder`` and ``CacheSettings`` embedder validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import semanticcache.embedders as embedders_mod
from semanticcache.config import CacheSettings
from semanticcache.embedders import get_embedder
from semanticcache.exceptions import NotSupportedEmbedderException


def test_cohere_embedder_raises_not_supported() -> None:
    """Cohere backend is not implemented yet."""
    settings = CacheSettings.model_validate({"embedder_type": "cohere"})
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


def test_voyage_embedder_constructed_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory passes model, dimensions, input_type hint, and API key."""
    captured: dict[str, object] = {}

    class _TrackingVoyage:
        """Capture constructor kwargs for factory wiring assertions."""

        def __init__(
            self,
            model_name: str = "",
            *,
            dimensions: int = 0,
            output_dimension: int | None = None,
            input_type: str | None = None,
            api_key: str | None = None,
        ) -> None:
            captured["model_name"] = model_name
            captured["dimensions"] = dimensions
            captured["output_dimension"] = output_dimension
            captured["input_type"] = input_type
            captured["api_key"] = api_key

    monkeypatch.setattr(embedders_mod, "VoyageEmbedder", _TrackingVoyage)
    settings = CacheSettings.model_validate(
        {
            "embedder_type": "voyage",
            "voyage_embedding_model": "voyage-4-lite",
            "voyage_embedding_dimensions": 512,
            "voyage_input_type": "document",
            "SEMANTIC_CACHE_VOYAGE_API_KEY": "vk",
        }
    )
    _ = get_embedder(settings)
    assert captured["model_name"] == "voyage-4-lite"
    assert captured["dimensions"] == 512
    assert captured["input_type"] == "document"
    assert captured["api_key"] == "vk"


def test_voyage_embedder_factory_uses_defaults_when_model_and_dims_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset voyage model and dimensions fall back to library defaults."""
    captured: dict[str, object] = {}

    class _TrackingVoyage:
        """Capture constructor kwargs for factory wiring assertions."""

        def __init__(
            self,
            model_name: str = "",
            *,
            dimensions: int = 0,
            input_type: str | None = None,
            api_key: str | None = None,
        ) -> None:
            captured["model_name"] = model_name
            captured["dimensions"] = dimensions
            captured["input_type"] = input_type
            captured["api_key"] = api_key

    monkeypatch.setattr(embedders_mod, "VoyageEmbedder", _TrackingVoyage)
    settings = CacheSettings.model_validate({"embedder_type": "voyage"})
    _ = get_embedder(settings)
    assert captured["model_name"] == embedders_mod.VOYAGE_DEFAULT_MODEL
    assert captured["dimensions"] == embedders_mod.VOYAGE_DEFAULT_DIMENSIONS
    assert captured["input_type"] is None
    assert captured["api_key"] is None
