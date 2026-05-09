"""Unit tests for ``OllamaEmbedder`` helpers (mocked optional deps)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from semanticcache.embedders import ollama as ollama_mod
from semanticcache.embedders.ollama import OllamaEmbedder
from semanticcache.exceptions import InvalidEmbeddingDimensionException


def test_require_openai_import_error_has_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing package surfaces an install extra hint."""
    import builtins

    real_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("simulated missing openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ImportError, match=r"fastapi-semcache\[embed-ollama\]"):
        OllamaEmbedder(model_name="m", dimensions=8)


def test_empty_model_name_raises() -> None:
    """Whitespace-only model id is rejected at construction."""
    with pytest.raises(ValueError, match="non-empty"):
        OllamaEmbedder(model_name="   ", dimensions=768)


def test_resolved_api_key_uses_placeholder_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted key becomes the documented placeholder string."""
    fake_openai = MagicMock()
    monkeypatch.setattr(ollama_mod, "_require_openai", lambda: fake_openai)

    emb = OllamaEmbedder(model_name="nomic-embed-text", dimensions=768)
    assert emb._resolved_api_key() == ollama_mod._PLACEHOLDER_API_KEY


def test_resolved_api_key_strips_user_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-empty user keys are trimmed for Bearer use."""
    fake_openai = MagicMock()
    monkeypatch.setattr(ollama_mod, "_require_openai", lambda: fake_openai)

    emb = OllamaEmbedder(
        model_name="x",
        dimensions=4,
        api_key="  secret  ",
    )
    assert emb._resolved_api_key() == "secret"


@pytest.mark.asyncio
async def test_embed_batch_validates_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectors shorter or longer than ``dimensions`` raise."""
    fake_openai = MagicMock()
    fake_openai.AsyncOpenAI = MagicMock()
    monkeypatch.setattr(ollama_mod, "_require_openai", lambda: fake_openai)

    emb = OllamaEmbedder(model_name="m", dimensions=2)
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(index=0, embedding=[1.0, 2.0, 3.0]),
    ]
    emb._client = MagicMock()
    emb._client.embeddings.create = AsyncMock(return_value=mock_response)

    with pytest.raises(InvalidEmbeddingDimensionException):
        await emb._embed_batch(["hello"])
