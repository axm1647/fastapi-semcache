"""Library-specific exceptions raised by semanticcache."""


class NotSupportedEmbedderException(Exception): ...


class EmbeddingDimensionException(Exception):
    """Base exception for embedding dimenstion issues"""


class EmbeddingDimensionUnavailableException(EmbeddingDimensionException):
    """Raised when an embedder cannot determine its output vector width."""


class InvalidEmbeddingDimensionException(EmbeddingDimensionException):
    """Raised when an embedder reports an invalid embedding dimension."""
