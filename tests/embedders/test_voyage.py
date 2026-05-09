"""Unit tests for ``VoyageEmbedder`` helpers (mocked optional deps)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semanticcache.embedders import voyage as voyage_mod
from semanticcache.embedders.voyage import VoyageEmbedder
from semanticcache.exceptions import InvalidEmbeddingDimensionException


def _make_embedder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name: str = "voyage-3",
    dimensions: int = 1024,
    output_dimension: int | None = None,
    input_type: str | None = None,
    api_key: str | None = "test-key",
) -> tuple[VoyageEmbedder, MagicMock, MagicMock]:
    """Construct a ``VoyageEmbedder`` with mocked voyageai and aiohttp.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        model_name: Model name for the embedder.
        dimensions: Expected embedding dimension.
        output_dimension: Optional API output_dimension param.
        input_type: Optional input_type hint.
        api_key: Voyage API key.

    Returns:
        Tuple of (embedder, fake_voyageai_module, fake_aiohttp_module).
    """
    fake_voyageai = MagicMock()
    fake_aiohttp = MagicMock()
    monkeypatch.setattr(voyage_mod, "_require_deps", lambda: (fake_voyageai, fake_aiohttp))

    emb = VoyageEmbedder(
        model_name=model_name,
        dimensions=dimensions,
        output_dimension=output_dimension,
        input_type=input_type,
        api_key=api_key,
    )
    return emb, fake_voyageai, fake_aiohttp


def test_require_deps_import_error_has_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing packages surface an install extra hint."""
    import builtins

    real_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("voyageai", "aiohttp"):
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ImportError, match=r"fastapi-semcache\[embed-voyage\]"):
        VoyageEmbedder()


def test_embedding_dim_returns_constructor_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embedding_dim`` reflects the ``dimensions`` constructor argument."""
    emb, _, _ = _make_embedder(monkeypatch, dimensions=512)
    assert emb.embedding_dim == 512


def test_cache_namespace_contains_model_and_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cache_namespace`` encodes backend, model, and dimension."""
    emb, _, _ = _make_embedder(monkeypatch, model_name="voyage-4-large", dimensions=2048)
    assert emb.cache_namespace == "voyage:voyage-4-large:2048"


def test_build_request_body_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Body excludes optional keys when ``input_type`` and ``output_dimension`` are None."""
    emb, _, _ = _make_embedder(monkeypatch)
    body = emb._build_request_body(["hello"])
    assert body == {"model": "voyage-3", "input": ["hello"]}
    assert "input_type" not in body
    assert "output_dimension" not in body


def test_build_request_body_with_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """``input_type`` is included in the body when set."""
    emb, _, _ = _make_embedder(monkeypatch, input_type="document")
    body = emb._build_request_body(["text"])
    assert body["input_type"] == "document"


def test_build_request_body_with_output_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """``output_dimension`` is included in the body when set."""
    emb, _, _ = _make_embedder(monkeypatch, output_dimension=256)
    body = emb._build_request_body(["text"])
    assert body["output_dimension"] == 256


def test_validate_token_counts_calls_voyageai_tokenize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_validate_token_counts`` delegates to ``voyageai.Client.tokenize``."""
    emb, fake_voyageai, _ = _make_embedder(monkeypatch, model_name="voyage-3")
    fake_enc_a = MagicMock()
    fake_enc_b = MagicMock()
    emb._vo_client.tokenize = MagicMock(return_value=[fake_enc_a, fake_enc_b])

    emb._validate_token_counts(["hello", "world"], offset=0)

    emb._vo_client.tokenize.assert_called_once_with(["hello", "world"], model="voyage-3")


def test_validate_token_counts_raises_on_encoding_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched tokenizer output count raises ``ValueError``."""
    emb, _, _ = _make_embedder(monkeypatch)
    emb._vo_client.tokenize = MagicMock(return_value=[MagicMock()])  # 1 for 2 texts

    with pytest.raises(ValueError, match="encodings"):
        emb._validate_token_counts(["a", "b"], offset=0)


@pytest.mark.asyncio
async def test_embed_returns_empty_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed([])`` returns ``[]`` without any API call."""
    emb, _, _ = _make_embedder(monkeypatch)
    result = await emb.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_batch_validates_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectors that don't match ``dimensions`` raise ``InvalidEmbeddingDimensionException``."""
    emb, _, _ = _make_embedder(monkeypatch, dimensions=4)
    emb._vo_client.tokenize = MagicMock(return_value=[MagicMock()])

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}],  # 3 != 4
        }
    )
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=mock_cm)
    emb._session = mock_session

    with pytest.raises(InvalidEmbeddingDimensionException):
        await emb.embed(["hello"])


@pytest.mark.asyncio
async def test_embed_batch_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-200 API status raises ``RuntimeError`` with status code."""
    emb, _, _ = _make_embedder(monkeypatch, dimensions=4)
    emb._vo_client.tokenize = MagicMock(return_value=[MagicMock()])

    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.text = AsyncMock(return_value="rate limited")
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=mock_cm)
    emb._session = mock_session

    with pytest.raises(RuntimeError, match="429"):
        await emb.embed(["hello"])


@pytest.mark.asyncio
async def test_embed_batch_raises_on_vector_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API returning wrong number of embeddings raises ``ValueError``."""
    emb, _, _ = _make_embedder(monkeypatch, dimensions=2)
    emb._vo_client.tokenize = MagicMock(return_value=[MagicMock(), MagicMock()])

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],  # 1 for 2 texts
        }
    )
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=mock_cm)
    emb._session = mock_session

    with pytest.raises(ValueError, match="1 embeddings for 2 inputs"):
        await emb.embed(["hello", "world"])


@pytest.mark.asyncio
async def test_embed_batch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful embed call returns correctly ordered vectors."""
    emb, _, _ = _make_embedder(monkeypatch, dimensions=2)
    emb._vo_client.tokenize = MagicMock(return_value=[MagicMock(), MagicMock()])

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "data": [
                {"index": 1, "embedding": [3.0, 4.0]},
                {"index": 0, "embedding": [1.0, 2.0]},
            ],
        }
    )
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=mock_cm)
    emb._session = mock_session

    result = await emb.embed(["first", "second"])
    assert result == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.asyncio
async def test_embed_batch_raises_on_duplicate_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate embedding indices in the API response raise ``ValueError``."""
    emb, _, _ = _make_embedder(monkeypatch, dimensions=2)
    emb._vo_client.tokenize = MagicMock(return_value=[MagicMock(), MagicMock()])

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "data": [
                {"index": 0, "embedding": [1.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ],
        }
    )
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.post = MagicMock(return_value=mock_cm)
    emb._session = mock_session

    with pytest.raises(ValueError, match="duplicate embedding index"):
        await emb.embed(["hello", "world"])
