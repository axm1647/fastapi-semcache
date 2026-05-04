"""Placeholder Cohere embedding backend (not wired into ``get_embedder`` yet)."""

from typing import override

from .openai import OpenAIEmbedder


class CohereEmbedder(OpenAIEmbedder):
    """Reserved embedding backend aligned with future Cohere API integration."""

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Raise until Cohere integration is implemented.

        Args:
            texts: Input strings (unused).

        Raises:
            NotImplementedError: Always, until this backend is completed.
        """
        raise NotImplementedError()
