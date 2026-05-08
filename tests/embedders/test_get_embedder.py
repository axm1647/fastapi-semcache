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
    ("cohere", "voyage", "ollama"),
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
