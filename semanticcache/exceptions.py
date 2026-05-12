"""Library-specific exceptions raised by semanticcache."""


class NotSupportedEmbedderException(Exception):
    """Raised when ``get_embedder`` is asked for an unsupported ``embedder_type``."""


class EmbeddingDimensionException(Exception):
    """Base exception for embedding dimension validation failures."""


class EmbeddingDimensionUnavailableException(EmbeddingDimensionException):
    """Raised when an embedder cannot determine its output vector width."""


class InvalidEmbeddingDimensionException(EmbeddingDimensionException):
    """Raised when an embedder reports an invalid embedding dimension."""


class NonFiniteEmbeddingValueException(Exception):
    """Raised when an embedding vector contains a non-finite value (nan or inf)."""

    def __init__(self, *, index: int, value: float) -> None:
        super().__init__(f"Non-finite value in embedding at index {index}: {value!r}")
        self.index = index
        self.value = value


class CacheTimeoutError(TimeoutError):
    """Raised when a cache dependency call exceeds the configured timeout."""

    def __init__(
        self,
        *,
        operation: str,
        timeout_seconds: float,
    ) -> None:
        """Initialize timeout metadata for logs and callers.

        Args:
            operation: Human-readable operation label (embed, db, redis).
            timeout_seconds: Applied timeout in seconds.
        """
        super().__init__(
            f"semantic cache operation timed out: {operation} after "
            f"{timeout_seconds:.3f}s"
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
