"""Integration tests for ``SBERTEmbedder`` (real sentence-transformers)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("sentence_transformers")

from semanticcache.embedders.sbert import SBERTEmbedder

# Default model: small, widely used; first run may download weights.
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EXPECTED_DIM = 384


def _load_embedder(**kwargs: object) -> SBERTEmbedder:
    """Construct ``SBERTEmbedder`` or skip when the model is unavailable.

    Args:
        **kwargs: Forwarded to ``SBERTEmbedder``.

    Returns:
        A loaded embedder instance.
    """
    try:
        return SBERTEmbedder(model_name=_DEFAULT_MODEL, **kwargs)
    except Exception as exc:
        pytest.skip(
            f"SBERT model unavailable (install embed-huggingface, cache weights, "
            f"or check network/proxy): {exc}"
        )


@pytest.fixture(scope="module")
def default_embedder() -> SBERTEmbedder:
    """Module-scoped default-model embedder."""
    return _load_embedder()


@pytest.fixture(scope="module")
def normalized_embedder() -> SBERTEmbedder:
    """Module-scoped embedder with L2 normalization enabled."""
    return _load_embedder(normalize_embeddings=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_dim_matches_known_model(
    default_embedder: SBERTEmbedder,
) -> None:
    """Loaded model reports the expected vector width."""
    assert default_embedder.embedding_dim == _EXPECTED_DIM


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_namespace_stable(default_embedder: SBERTEmbedder) -> None:
    """Namespace string matches backend conventions for the default model."""
    assert (
        default_embedder.cache_namespace
        == f"sbert:{_DEFAULT_MODEL}:{_EXPECTED_DIM}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_shape_and_finite(default_embedder: SBERTEmbedder) -> None:
    """``embed`` returns one finite vector per string, correct length."""
    texts = ["hello semantic cache", "second phrase"]
    vectors = await default_embedder.embed(texts)
    assert len(vectors) == len(texts)
    for row in vectors:
        assert len(row) == _EXPECTED_DIM
        assert all(math.isfinite(x) for x in row)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normalize_embeddings_l2_unit(
    normalized_embedder: SBERTEmbedder,
) -> None:
    """With default settings, rows are L2-normalized (cosine-ready)."""
    vectors = await normalized_embedder.embed(["normalization check"])
    assert len(vectors) == 1
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5)
