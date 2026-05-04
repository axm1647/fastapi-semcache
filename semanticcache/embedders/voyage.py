"""Placeholder Voyage embedding backend (not wired into ``get_embedder`` yet)."""

from typing import override

from ._base import BaseEmbedder


class VoyageEmbedder(BaseEmbedder):
    """Reserved Voyage embedder; remains abstract until properties and ``embed`` exist."""

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Raise until Voyage integration is implemented.

        Args:
            texts: Input strings (unused).

        Raises:
            NotImplementedError: Always, until this backend is completed.
        """
        raise NotImplementedError()
