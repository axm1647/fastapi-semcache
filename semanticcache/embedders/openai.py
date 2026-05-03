from typing import override

from ._base import BaseEmbedder


def _require_openai():
    """Import openai or raise with install hint.

    Returns:
        The  class.

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
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        _normalize_embeddings: bool = True,
    ) -> None:
        """
        Embed text with hosted openai embedding model.

        Defaults to ``text-embedding-3-small``
        OpenAI embedding models don't expose a normalize param so this doesn't do anything.
        It is kept to stay consistent with SBERTEmbedder()
        """
        _ = _require_openai()
        raise NotImplementedError()

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()
