"""Library-specific exceptions raised by semanticcache."""


class NotSupportedEmbedderException(Exception):
    """Raised when ``get_embedder`` is asked for an unsupported ``embedder_type``."""


class EmbeddingDimensionException(Exception):
    """Base exception for embedding dimension validation failures."""


class EmbeddingDimensionUnavailableException(EmbeddingDimensionException):
    """Raised when an embedder cannot determine its output vector width."""


class InvalidEmbeddingDimensionException(EmbeddingDimensionException):
    """Raised when an embedder reports an invalid embedding dimension."""
