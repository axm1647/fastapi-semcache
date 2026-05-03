"""Pluggable embedding backends."""

# pyright: reportImplicitStringConcatenation=false

from semanticcache.config import CacheSettings, get_cache_settings
from semanticcache.embedders.base import BaseEmbedder
from semanticcache.embedders.sbert import SBERTEmbedder


def get_embedder(settings: CacheSettings | None = None) -> BaseEmbedder:
    """Construct an embedder from application settings.

    Args:
        settings: Cache settings; defaults to ``get_cache_settings()``.

    Returns:
        A ``BaseEmbedder`` instance.

    Raises:
        NotImplementedError: If ``embedder_type`` is not supported yet.
    """
    resolved = settings if settings is not None else get_cache_settings()
    if resolved.embedder_type == "local":
        return SBERTEmbedder()
    if resolved.embedder_type == "openai":
        raise NotImplementedError(
            "OpenAI embedder is not implemented yet; use embedder_type='local' "
            "or install and wire OpenAIEmbedder when available."
        )


__all__ = [
    "BaseEmbedder",
    "SBERTEmbedder",
    "get_embedder",
]
