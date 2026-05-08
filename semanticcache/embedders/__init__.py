"""Pluggable embedding backends."""

# pyright: reportImplicitStringConcatenation=false

from ..config import CacheSettings, get_cache_settings
from ..exceptions import NotSupportedEmbedderException
from ._base import BaseEmbedder
from .openai import OpenAIEmbedder
from .sbert import SBERTEmbedder


def get_embedder(settings: CacheSettings | None = None) -> BaseEmbedder:
    """Construct an embedder from application settings.

    Args:
        settings: Cache settings; defaults to ``get_cache_settings()``.

    Returns:
        A ``BaseEmbedder`` instance.

    Raises:
        NotSupportedEmbedderException: If ``embedder_type`` is not supported.
    """
    resolved = settings if settings is not None else get_cache_settings()
    if resolved.embedder_type == "huggingface":
        return SBERTEmbedder(token=resolved.hugging_face_api_key)
    if resolved.embedder_type == "openai":
        return OpenAIEmbedder()
    if resolved.embedder_type == "cohere":
        raise NotSupportedEmbedderException("Cohere embeddings are not supported yet.")
    if resolved.embedder_type == "voyage":
        raise NotSupportedEmbedderException("Voyage embeddings are not supported yet.")
    if resolved.embedder_type == "ollama":
        raise NotSupportedEmbedderException("Ollama embeddings are not supported yet.")
    raise NotSupportedEmbedderException(
        "This embeddings option is not supported. "
        "Please check README for available embedding options."
    )


__all__: list[str] = [
    "BaseEmbedder",
    "SBERTEmbedder",
    "get_embedder",
    "OpenAIEmbedder",
]
