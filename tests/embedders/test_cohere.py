"""Unit tests for ``CohereEmbedder`` helpers (mocked optional deps)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from semanticcache.embedders import cohere as cohere_mod
from semanticcache.embedders.cohere import CohereEmbedder
from semanticcache.exceptions import InvalidEmbeddingDimensionException


def _make_embedder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name: str = "embed-v4.0",
    dimensions: int = 1536,
    input_type: str = "search_document",
    output_dimension: int | None = None,
    truncate: str | None = None,
    api_key: str | None = "test-key",
) -> tuple[CohereEmbedder, MagicMock]:
    """Construct a ``CohereEmbedder`` with mocked cohere module.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        model_name: Model name for the embedder.
        dimensions: Expected embedding dimension.
        input_type: Cohere input_type hint.
        output_dimension: Optional v2 output_dimension param.
        truncate: Optional truncate mode.
        api_key: Cohere API key.

    Returns:
        Tuple of (embedder, fake_cohere_module).
    """
    fake_cohere = MagicMock()
    monkeypatch.setattr(
        cohere_mod,
        "_require_cohere",
        lambda: fake_cohere,
    )

    emb = CohereEmbedder(
        model_name=model_name,
        dimensions=dimensions,
        input_type=input_type,  # type: ignore[arg-type]
        output_dimension=output_dimension,
        truncate=truncate,  # type: ignore[arg-type]
        api_key=api_key,
    )
    return emb, fake_cohere


def test_require_cohere_import_error_has_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing package surfaces an install extra hint."""
    import builtins

    real_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cohere":
            raise ImportError("simulated missing cohere")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ImportError, match=r"fastapi-semcache\[embed-cohere\]"):
        CohereEmbedder()


def test_embedding_dim_returns_constructor_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embedding_dim`` reflects the ``dimensions`` constructor argument."""
    emb, _ = _make_embedder(monkeypatch, dimensions=512)
    assert emb.embedding_dim == 512


def test_cache_namespace_contains_model_and_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cache_namespace`` encodes backend, model, and dimension."""
    emb, _ = _make_embedder(
        monkeypatch, model_name="embed-multilingual-v3.0", dimensions=1024
    )
    assert emb.cache_namespace == "cohere:embed-multilingual-v3.0:1024"


def test_build_v2_embed_kwargs_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    """v2 kwargs exclude optional keys when unset."""
    emb, _ = _make_embedder(monkeypatch)
    body = emb._build_v2_embed_kwargs(["hello"])
    assert body == {
        "texts": ["hello"],
        "model": "embed-v4.0",
        "input_type": "search_document",
        "embedding_types": ["float"],
    }


def test_build_v2_embed_kwargs_with_output_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``output_dimension`` and ``truncate`` are included when set."""
    emb, _ = _make_embedder(
        monkeypatch,
        output_dimension=512,
        truncate="END",
        input_type="search_query",
    )
    body = emb._build_v2_embed_kwargs(["text"])
    assert body["output_dimension"] == 512
    assert body["truncate"] == "END"
    assert body["input_type"] == "search_query"


def test_extract_float_vectors_from_embeddings_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level float response type is parsed."""
    response = MagicMock()
    response.response_type = "embeddings_floats"
    response.embeddings = [[1.0, 2.0], [3.0, 4.0]]
    emb, _ = _make_embedder(monkeypatch, dimensions=2)
    assert emb._extract_float_vectors(response) == [[1.0, 2.0], [3.0, 4.0]]


def test_extract_float_vectors_from_by_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """v2 by-type response reads ``embeddings.float_``."""
    floats = MagicMock()
    floats.float_ = [[1.0], [2.0]]
    response = MagicMock()
    response.embeddings = floats
    emb, _ = _make_embedder(monkeypatch, dimensions=1)
    assert emb._extract_float_vectors(response) == [[1.0], [2.0]]


@pytest.mark.asyncio
async def test_embed_returns_empty_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed([])`` returns ``[]`` without any API call."""
    emb, fake_cohere = _make_embedder(monkeypatch)
    result = await emb.embed([])
    assert result == []
    fake_cohere.AsyncClient.assert_not_called()


@pytest.mark.asyncio
async def test_embed_via_top_level_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default path uses ``AsyncClient.embed`` with SDK batching."""
    emb, fake_cohere = _make_embedder(monkeypatch, dimensions=2)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.response_type = "embeddings_floats"
    mock_response.embeddings = [[1.0, 2.0], [3.0, 4.0]]
    mock_client.embed = AsyncMock(return_value=mock_response)
    fake_cohere.AsyncClient.return_value = mock_client
    emb._client = mock_client

    result = await emb.embed(["a", "b"])
    assert result == [[1.0, 2.0], [3.0, 4.0]]
    mock_client.embed.assert_awaited_once()
    call_kwargs = mock_client.embed.await_args.kwargs
    assert call_kwargs["batching"] is True
    assert call_kwargs["embedding_types"] == ["float"]


@pytest.mark.asyncio
async def test_embed_via_v2_when_output_dimension_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``output_dimension`` selects manual v2 batching."""
    emb, fake_cohere = _make_embedder(monkeypatch, dimensions=2, output_dimension=512)
    mock_client = MagicMock()
    mock_v2 = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.float_ = [[1.0, 2.0]]
    mock_response = MagicMock()
    mock_response.embeddings = mock_embeddings
    mock_v2.embed = AsyncMock(return_value=mock_response)
    mock_client.v2 = mock_v2
    fake_cohere.AsyncClient.return_value = mock_client
    emb._client = mock_client

    result = await emb.embed(["hello"])
    assert result == [[1.0, 2.0]]
    mock_v2.embed.assert_awaited_once()
    mock_client.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embed_raises_on_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectors that don't match ``dimensions`` raise ``InvalidEmbeddingDimensionException``."""
    emb, _ = _make_embedder(monkeypatch, dimensions=4)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.response_type = "embeddings_floats"
    mock_response.embeddings = [[1.0, 2.0, 3.0]]
    mock_client.embed = AsyncMock(return_value=mock_response)
    emb._client = mock_client

    with pytest.raises(InvalidEmbeddingDimensionException):
        await emb.embed(["hello"])


@pytest.mark.asyncio
async def test_embed_raises_on_vector_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API returning wrong number of embeddings raises ``ValueError``."""
    emb, _ = _make_embedder(monkeypatch, dimensions=2)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.response_type = "embeddings_floats"
    mock_response.embeddings = [[1.0, 2.0]]
    mock_client.embed = AsyncMock(return_value=mock_response)
    emb._client = mock_client

    with pytest.raises(ValueError, match="1 embeddings for 2 inputs"):
        await emb.embed(["hello", "world"])


@pytest.mark.asyncio
async def test_aclose_closes_httpx_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """``aclose`` awaits the underlying httpx client close."""
    emb, fake_cohere = _make_embedder(monkeypatch)
    mock_httpx = MagicMock()
    mock_httpx.aclose = AsyncMock()
    mock_wrapper = MagicMock()
    mock_wrapper.httpx_client.httpx_client = mock_httpx
    mock_client = MagicMock()
    mock_client._client_wrapper = mock_wrapper
    emb._client = mock_client

    await emb.aclose()

    mock_httpx.aclose.assert_awaited_once()
    assert emb._client is None
    fake_cohere.AsyncClient.assert_not_called()
