"""Integration tests for ``SBERTEmbedder`` (real sentence-transformers)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("sentence_transformers")

from semanticcache.embedders.sbert import SBERTEmbedder

# Default model: small, widely used; first run may download weights.
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EXPECTED_DIM = 384


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_dim_matches_known_model() -> None:
    """Loaded model reports the expected vector width."""
    embedder = SBERTEmbedder(model_name=_DEFAULT_MODEL)
    assert embedder.embedding_dim == _EXPECTED_DIM


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_namespace_stable() -> None:
    """Namespace string matches backend conventions for the default model."""
    embedder = SBERTEmbedder(model_name=_DEFAULT_MODEL)
    assert embedder.cache_namespace == f"sbert:{_DEFAULT_MODEL}:{_EXPECTED_DIM}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_shape_and_finite() -> None:
    """``embed`` returns one finite vector per string, correct length."""
    embedder = SBERTEmbedder(model_name=_DEFAULT_MODEL)
    texts = ["hello semantic cache", "second phrase"]
    vectors = await embedder.embed(texts)
    assert len(vectors) == len(texts)
    for row in vectors:
        assert len(row) == _EXPECTED_DIM
        assert all(math.isfinite(x) for x in row)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normalize_embeddings_l2_unit() -> None:
    """With default settings, rows are L2-normalized (cosine-ready)."""
    embedder = SBERTEmbedder(model_name=_DEFAULT_MODEL, normalize_embeddings=True)
    vectors = await embedder.embed(["normalization check"])
    assert len(vectors) == 1
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5)
