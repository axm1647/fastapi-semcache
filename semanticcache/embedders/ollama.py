"""Placeholder Ollama embedding backend (not wired into ``get_embedder`` yet)."""

from typing import override

from ._base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Reserved Ollama embedder; remains abstract until properties and ``embed`` exist."""

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Raise until Ollama integration is implemented.

        Args:
            texts: Input strings (unused).

        Raises:
            NotImplementedError: Always, until this backend is completed.
        """
        raise NotImplementedError()
