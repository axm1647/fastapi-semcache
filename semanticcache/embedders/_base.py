"""Abstract embedder interface for semantic caching."""

from abc import ABC, abstractmethod

from ..exceptions import InvalidEmbeddingDimensionException


class BaseEmbedder(ABC):
    """Abstract base class for pluggable text embedding backends."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return dense vector length produced by ``embed``."""

    @property
    @abstractmethod
    def cache_namespace(self) -> str:
        """Return a stable id for this model configuration (storage namespacing)."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Turn input strings into embedding vectors.

        Args:
            texts: Batch of strings to embed. Order is preserved in the output.

        Returns:
            One embedding vector per input string, same length and order as texts.
            Each inner list is a dense floating-point vector (model-dependent length).
        """

    @staticmethod
    def require_positive_dim(dim: int) -> int:
        if dim < 1:
            msg = "embedding_dim must be positive"
            raise InvalidEmbeddingDimensionException(msg)
        return dim
