"""Pluggable embedding backends."""

# pyright: reportImplicitStringConcatenation=false

from ..config import CacheSettings, get_cache_settings
from ..exceptions import NotSupportedEmbedderException
from ._base import BaseEmbedder
from .ollama import OllamaEmbedder
from .openai import OpenAIEmbedder
from .sbert import SBERTEmbedder
from .voyage import VOYAGE_DEFAULT_DIMENSIONS, VOYAGE_DEFAULT_MODEL, VoyageEmbedder


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
        return SBERTEmbedder(api_key=resolved.hugging_face_api_key)
    if resolved.embedder_type == "openai":
        return OpenAIEmbedder(api_key=resolved.openai_api_key)
    if resolved.embedder_type == "cohere":
        raise NotSupportedEmbedderException("Cohere embeddings are not supported yet.")
    if resolved.embedder_type == "voyage":
        return VoyageEmbedder(
            model_name=resolved.voyage_embedding_model or VOYAGE_DEFAULT_MODEL,
            dimensions=resolved.voyage_embedding_dimensions
            or VOYAGE_DEFAULT_DIMENSIONS,
            input_type=resolved.voyage_input_type,
            api_key=resolved.voyage_api_key,
        )
    if resolved.embedder_type == "ollama":
        model: str | None = resolved.ollama_embedding_model
        dims: int | None = resolved.ollama_embedding_dimensions
        if model is None or dims is None:
            raise ValueError(
                "ollama_embedding_model and ollama_embedding_dimensions are required"
            )
        return OllamaEmbedder(
            model_name=model,
            dimensions=dims,
            api_key=resolved.ollama_api_key,
            base_url=resolved.ollama_base_url,
        )
    raise NotSupportedEmbedderException(
        "This embeddings option is not supported. "
        "Please check README for available embedding options."
    )


__all__: list[str] = [
    "BaseEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "SBERTEmbedder",
    "VoyageEmbedder",
    "get_embedder",
]
