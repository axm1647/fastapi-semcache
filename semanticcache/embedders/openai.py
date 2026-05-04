from typing import override

from ..exceptions import InvalidEmbeddingDimensionException
from ._base import BaseEmbedder


def _require_openai():
    """Import openai or raise with install hint.

    Returns:
        Tuple of the ``openai`` and ``tiktoken`` modules.

    Raises:
        ImportError: If openai and tiktoken are not installed.
    """
    try:
        import openai as AI
        import tiktoken as TK
    except ImportError as exc:
        _missing = (
            "OpenAIEmbedder requires optional dependencies. pip install 'embed-openai'."
        )
        raise ImportError(_missing) from exc
    return AI, TK


class OpenAIEmbedder(BaseEmbedder):
    """Embed text with a hosted OpenAI embedding model (not yet implemented)."""

    _model_name: str
    _dimensions: int
    _normalize_embeddings: bool

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        dimensions: int = 1536,
    ) -> None:
        """Declare OpenAI embedding settings for dimension and table namespacing.

        Args:
            model_name: Embedding model id passed to the OpenAI API.
            dimensions: Requested embedding width (defaults to 1536 for text-embedding-3-small).
                This is passed to the OpenAI API and used for table namespacing + column.
        """
        _ = _require_openai()
        if dimensions < 1:
            msg = "dimensions must be positive"
            raise InvalidEmbeddingDimensionException(msg)
        self._model_name = model_name
        self._dimensions = dimensions

    @property
    @override
    def embedding_dim(self) -> int:
        """Return the configured embedding width.

        Returns:
            Value of ``dimensions`` passed at construction.
        """
        return self._dimensions

    @property
    @override
    def cache_namespace(self) -> str:
        """Return a stable namespace for pgvector and Redis namespacing.

        Returns:
            Identifier derived from backend, model id, and dimension.
        """
        return f"openai:{self._model_name}:{self._dimensions}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()
