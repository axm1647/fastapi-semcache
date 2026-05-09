"""Ollama embedding backend via OpenAI-compatible ``/v1/embeddings`` (async client)."""

from __future__ import annotations

import threading
from types import ModuleType

# openai typing is incomplete; keep runtime behavior explicit.
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from typing import Any, override, TYPE_CHECKING

from ..exceptions import InvalidEmbeddingDimensionException
from ._base import BaseEmbedder

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_DEFAULT_LOCAL_BASE = "http://127.0.0.1:11434/v1"
"""OpenAI-compatible root URL for a default local Ollama install."""

_MAX_INPUTS_PER_REQUEST = 512
"""Batch size per HTTP request (conservative for local inference memory)."""

_PLACEHOLDER_API_KEY = "ollama"
"""Bearer value when no API key is configured (avoids picking up ``OPENAI_API_KEY``)."""


def _require_openai() -> ModuleType:
    """Import ``openai`` or raise with install hint.

    Returns:
        The ``openai`` module.

    Raises:
        ImportError: If ``openai`` is not installed.
    """
    try:
        import openai
    except ImportError as exc:
        _missing = (
            "OllamaEmbedder requires optional dependencies. "
            "pip install 'fastapi-semcache[embed-ollama]'."
        )
        raise ImportError(_missing) from exc
    return openai


class OllamaEmbedder(BaseEmbedder):
    """Embed text using Ollama's OpenAI-compatible embeddings API."""

    _model_name: str
    _dimensions: int
    _openai: ModuleType
    _api_key: str | None
    _base_url: str
    _client: Any | None
    _client_lock: threading.Lock

    def __init__(
        self,
        model_name: str,
        *,
        dimensions: int,
        api_key: str | None = None,
        base_url: str = _DEFAULT_LOCAL_BASE,
    ) -> None:
        """Configure the async client and declared vector width for storage.

        Args:
            model_name: Model id passed to ``embeddings.create`` (your running Ollama
                embedding model). Must match the vector width given by ``dimensions``.
            dimensions: Expected embedding length for validation and storage. Must
                match the model output and pgvector column dimension.
            api_key: Optional Bearer token when Ollama is configured with auth.
                When omitted, a placeholder key is sent so the SDK does not read
                ``OPENAI_API_KEY`` from the environment.
            base_url: OpenAI-compatible API root, including the ``/v1`` path (for
                example ``http://127.0.0.1:11434/v1``).
        """
        stripped_model: str = model_name.strip()
        if not stripped_model:
            msg: str = "model_name must be a non-empty string"
            raise ValueError(msg)

        openai_mod: ModuleType = _require_openai()
        self._dimensions = BaseEmbedder.require_positive_dim(dimensions)
        self._model_name = stripped_model
        self._openai = openai_mod
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = None
        self._client_lock = threading.Lock()

    def _resolved_api_key(self) -> str:
        """Return the Bearer token for ``AsyncOpenAI``.

        Returns:
            User-supplied key or a placeholder for unsecured local servers.
        """
        if self._api_key is not None and self._api_key.strip():
            return self._api_key.strip()
        return _PLACEHOLDER_API_KEY

    def _get_client(self) -> "AsyncOpenAI":
        """Return a lazily constructed ``AsyncOpenAI`` client.

        Thread-safe so concurrent ``embed`` tasks share one client.

        Returns:
            Shared async OpenAI client instance targeting Ollama.
        """
        with self._client_lock:
            if self._client is None:
                self._client = self._openai.AsyncOpenAI(
                    api_key=self._resolved_api_key(),
                    base_url=self._base_url,
                )
            return self._client  # type: ignore

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for one API batch.

        Args:
            texts: Non-empty batch of strings.

        Returns:
            One vector per string, same order as ``texts``.

        Raises:
            ValueError: If the API response count, indices, or duplicates do not match
                ``texts``.
            InvalidEmbeddingDimensionException: If a vector length does not match
                ``dimensions``.
        """
        create_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }
        response = await self._get_client().embeddings.create(**create_kwargs)
        rows = response.data
        if len(rows) != len(texts):
            msg = f"embeddings API returned {len(rows)} vectors for {len(texts)} inputs"
            raise ValueError(msg)
        by_index: dict[int, Any] = {}
        for row in rows:
            idx = row.index
            if idx in by_index:
                msg = f"duplicate embedding index {idx} in API response"
                raise ValueError(msg)
            by_index[idx] = row
        expected = set(range(len(texts)))
        if set(by_index.keys()) != expected:
            msg: str = (
                f"embedding indices {sorted(by_index.keys())!r} do not match "
                f"inputs 0..{len(texts) - 1}"
            )
            raise ValueError(msg)

        out: list[list[float]] = []
        for i in range(len(texts)):
            vec = list(by_index[i].embedding)
            if len(vec) != self._dimensions:
                msg = (
                    f"expected embedding length {self._dimensions}, got {len(vec)} "
                    f"for model {self._model_name!r}"
                )
                raise InvalidEmbeddingDimensionException(msg)
            out.append(vec)
        return out

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
        return f"ollama:{self._model_name}:{self._dimensions}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama's OpenAI-compatible embeddings endpoint in chunks.

        Args:
            texts: Strings to embed.

        Returns:
            Embedding vectors as nested lists, aligned with ``texts``.
        """
        if not texts:
            return []

        results: list[list[float]] = []
        for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
            results.extend(await self._embed_batch(batch))
        return results
