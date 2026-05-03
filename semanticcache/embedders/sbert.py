"""Local Sentence-BERT embedding backend (sentence-transformers)."""

# sentence-transformers typing is incomplete; interactions stay runtime-checked.
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import asyncio
from typing import cast, final, override

from ._base import BaseEmbedder


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
            "'semanticcache-py[embed-local-cpu]'. GPU (CUDA PyTorch): pip install "
            "'semanticcache-py[embed-local-gpu]' with --extra-index-url from "
            "https://pytorch.org/get-started/locally/ (see README). Shorthand for "
            "CPU: pip install 'semanticcache-py[embed-local]'."
        )
        raise ImportError(_missing) from exc
    return ST


@final
class SBERTEmbedder(BaseEmbedder):
    """Embed text using a local sentence-transformers model.

    Defaults to ``all-MiniLM-L6-v2`` (384-dimensional), matching the pgvector
    schema in ``docker/init.sql``.
    """

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
        self._model = SentenceTransformer(model_name)
        self._normalize_embeddings: bool = normalize_embeddings

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
