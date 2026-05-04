"""Local Sentence-BERT embedding backend (sentence-transformers)."""

# sentence-transformers typing is incomplete; interactions stay runtime-checked.
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import asyncio
from typing import cast, final, override

from ..exceptions import (
    EmbeddingDimensionUnavailableException,
    InvalidEmbeddingDimensionException,
)
from ..config import get_cache_settings
from ._base import BaseEmbedder


def _require_positive_dim(dim: int) -> int:
    """Validate embedding dimension for storage alignment.

    Args:
        dim: Declared vector length.

    Returns:
        Same value when valid.

    Raises:
        ValueError: If ``dim`` is not positive.
    """
    if dim < 1:
        msg = "embedding_dim must be positive"
        raise InvalidEmbeddingDimensionException(msg)
    return dim


def _require_sentence_transformers():
    """Import sentence-transformers or raise with install hint.

    Returns:
        The SentenceTransformer class.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer as ST
    except ImportError as exc:
        _missing = (
            "SBERTEmbedder requires optional dependencies. CPU: pip install "
            "'fastapi-semcache[embed-huggingface-cpu]'. GPU (CUDA PyTorch): pip install "
            "'fastapi-semcache[embed-huggingface-gpu]' with --extra-index-url from "
            "https://pytorch.org/get-started/locally/ (see README). Shorthand for "
            "CPU: pip install 'fastapi-semcache[embed-huggingface]'."
        )
        raise ImportError(_missing) from exc
    return ST


@final
class SBERTEmbedder(BaseEmbedder):
    """Embed text using a local sentence-transformers model.

    Defaults to ``all-MiniLM-L6-v2`` (384-dimensional). Vector tables are created at
    runtime from ``embedding_dim`` and ``cache_namespace``.
    """

    _model_name: str

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        normalize_embeddings: bool = True,
    ) -> None:
        """Load a SentenceTransformer model.

        Args:
            model_name: Hugging Face model id or local path passed to
                ``SentenceTransformer``.
            normalize_embeddings: When True, L2-normalize vectors (recommended for
                cosine similarity with pgvector ``vector_cosine_ops``).
        """
        SentenceTransformer = _require_sentence_transformers()
        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name, token=get_cache_settings().hugging_face_api_key
        )  # optional Hugging Face API key for private models and rate limiting
        self._normalize_embeddings: bool = normalize_embeddings

    @property
    @override
    def embedding_dim(self) -> int:
        """Return the model output dimension.

        Returns:
            Embedding width from the loaded SentenceTransformer.

        Raises:
            EmbeddingDimensionUnavailableException: If the model did not report a
                dimension.
        """
        dim = self._model.get_embedding_dimension()
        if dim is None:
            msg = "SentenceTransformer did not report an embedding dimension"
            raise EmbeddingDimensionUnavailableException(msg)
        return _require_positive_dim(int(dim))

    @property
    @override
    def cache_namespace(self) -> str:
        """Return a stable namespace for pgvector and Redis namespacing.

        Returns:
            Identifier derived from backend, model id, and dimension.
        """
        return f"sbert:{self._model_name}:{self.embedding_dim}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts on a worker thread to avoid blocking the event loop.

        Args:
            texts: Strings to embed.

        Returns:
            Embedding vectors as nested lists, aligned with ``texts``.
        """
        if not texts:
            return []

        def _encode() -> list[list[float]]:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return cast(list[list[float]], embeddings.tolist())

        return await asyncio.to_thread(_encode)
