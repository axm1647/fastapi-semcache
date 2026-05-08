"""Unit tests for ``SBERTEmbedder`` and SBERT helpers (mocked, no model weights)."""

# Tests intentionally touch private helpers and ``_model`` for behavior checks.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from semanticcache.embedders import sbert as sbert_mod
from semanticcache.embedders.sbert import (
    BaseEmbedder,
    SBERTEmbedder,
    _require_sentence_transformers,
)
from semanticcache.exceptions import (
    EmbeddingDimensionUnavailableException,
    InvalidEmbeddingDimensionException,
)


@pytest.mark.parametrize(
    ("dim", "expected"),
    [(1, 1), (384, 384)],
)
def test_require_positive_dim_accepts_positive(dim: int, expected: int) -> None:
    """Return the same value when the dimension is positive."""
    assert BaseEmbedder.require_positive_dim(dim) == expected


@pytest.mark.parametrize("dim", [0, -1])
def test_require_positive_dim_rejects_non_positive(dim: int) -> None:
    """Reject zero or negative embedding widths."""
    with pytest.raises(InvalidEmbeddingDimensionException, match="positive"):
        BaseEmbedder.require_positive_dim(dim)


def test_require_sentence_transformers_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface an install hint when sentence-transformers is missing."""
    import builtins

    real_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_hook)
    with pytest.raises(ImportError, match=r"fastapi-semcache\[embed-huggingface"):
        _require_sentence_transformers()


class _FakeSentenceTransformer:
    """Minimal stand-in for ``SentenceTransformer`` instances."""

    def __init__(
        self,
        model_name: str,
        *,
        token: str | None = None,
        embedding_dim: int | None = 384,
        encode_result: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    ) -> None:
        """Capture ctor args and configure fake encode output.

        Args:
            model_name: Recorded model id (same as real API).
            token: Hugging Face token passed through from settings.
            embedding_dim: Value returned by ``get_embedding_dimension`` (or ``None``).
            encode_result: Fixed array from ``encode``; default is zeros.
        """
        self.model_name = model_name
        self.token = token
        self._embedding_dim = embedding_dim
        self._encode_result = encode_result
        self.encode_calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int | None:
        """Return configured width or ``None`` for error-path tests."""
        return self._embedding_dim

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Record kwargs and return a deterministic numpy array."""
        self.encode_calls.append(
            {
                "texts": list(texts),
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        if self._encode_result is not None:
            return self._encode_result
        n = len(texts)
        d = int(self._embedding_dim) if self._embedding_dim is not None else 384
        return np.zeros((n, d), dtype=np.float64)


def _patch_require_st(monkeypatch: pytest.MonkeyPatch, cls: type[Any]) -> None:
    """Make ``SBERTEmbedder`` load ``cls`` instead of real ``SentenceTransformer``."""
    monkeypatch.setattr(sbert_mod, "_require_sentence_transformers", lambda: cls)


@pytest.mark.asyncio
async def test_embed_empty_skips_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty batch returns immediately without calling ``encode``."""
    _patch_require_st(monkeypatch, _FakeSentenceTransformer)
    embedder = SBERTEmbedder()
    fake = embedder._model
    assert isinstance(fake, _FakeSentenceTransformer)
    out = await embedder.embed([])
    assert out == []
    assert fake.encode_calls == []


@pytest.mark.asyncio
async def test_embed_batches_and_tolist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-empty input yields nested lists aligned with ``texts``."""
    _patch_require_st(monkeypatch, _FakeSentenceTransformer)
    embedder = SBERTEmbedder()
    texts = ["a", "b"]
    out = await embedder.embed(texts)
    fake = embedder._model
    assert isinstance(fake, _FakeSentenceTransformer)
    assert len(out) == 2
    assert len(out[0]) == 384
    assert fake.encode_calls[0]["texts"] == texts
    assert fake.encode_calls[0]["normalize_embeddings"] is True
    assert fake.encode_calls[0]["convert_to_numpy"] is True
    assert fake.encode_calls[0]["show_progress_bar"] is False


@pytest.mark.asyncio
async def test_embed_respects_normalize_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``normalize_embeddings=False`` is forwarded to ``encode``."""
    _patch_require_st(monkeypatch, _FakeSentenceTransformer)
    embedder = SBERTEmbedder(normalize_embeddings=False)
    await embedder.embed(["x"])
    fake = embedder._model
    assert isinstance(fake, _FakeSentenceTransformer)
    assert fake.encode_calls[0]["normalize_embeddings"] is False


@pytest.mark.asyncio
async def test_embed_uses_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Heavy encode work is scheduled with ``asyncio.to_thread``."""
    _patch_require_st(monkeypatch, _FakeSentenceTransformer)
    embedder = SBERTEmbedder()

    async def fake_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(sbert_mod.asyncio, "to_thread", fake_to_thread)
    out = await embedder.embed(["only"])
    assert len(out) == 1


def test_init_passes_huggingface_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provided HF token is passed to the transformer constructor."""
    holder: dict[str, str | None] = {}

    class _TrackingFake(_FakeSentenceTransformer):
        """Record ``token`` from the SentenceTransformer-style constructor."""

        def __init__(
            self, model_name: str, *, token: str | None = None, **kwargs: Any
        ) -> None:
            holder["token"] = token
            super().__init__(model_name, token=token, **kwargs)

    _patch_require_st(monkeypatch, _TrackingFake)
    SBERTEmbedder(api_key="hf-test-key")
    assert holder["token"] == "hf-test-key"


def test_cache_namespace_includes_model_and_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace is stable and encodes backend, model id, and width."""
    _patch_require_st(monkeypatch, _FakeSentenceTransformer)
    name = "org/custom-model"
    embedder = SBERTEmbedder(model_name=name)
    assert embedder.cache_namespace == f"sbert:{name}:384"


def test_embedding_dim_none_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing dimension from the model raises a dedicated exception."""

    class _NoDim(_FakeSentenceTransformer):
        """Act like a model that does not report output width."""

        def __init__(
            self, model_name: str, *, token: str | None = None, **kwargs: Any
        ) -> None:
            super().__init__(model_name, token=token, embedding_dim=None, **kwargs)

    _patch_require_st(monkeypatch, _NoDim)
    embedder = SBERTEmbedder()
    with pytest.raises(EmbeddingDimensionUnavailableException, match="dimension"):
        _ = embedder.embedding_dim


def test_embedding_dim_non_positive_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-positive reported width is rejected like other backends."""

    class _BadDim(_FakeSentenceTransformer):
        """Act like a model that reports an invalid vector width."""

        def __init__(
            self, model_name: str, *, token: str | None = None, **kwargs: Any
        ) -> None:
            super().__init__(model_name, token=token, embedding_dim=0, **kwargs)

    _patch_require_st(monkeypatch, _BadDim)
    embedder = SBERTEmbedder()
    with pytest.raises(InvalidEmbeddingDimensionException, match="positive"):
        _ = embedder.embedding_dim
