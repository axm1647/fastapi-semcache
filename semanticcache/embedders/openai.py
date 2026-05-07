"""OpenAI API embedding backend (``openai`` async client)."""

from __future__ import annotations
import threading
from types import ModuleType

# openai and tiktoken typing is incomplete; keep runtime behavior explicit.
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from typing import Any, override, TYPE_CHECKING

from ..exceptions import InvalidEmbeddingDimensionException
from ._base import BaseEmbedder

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_MAX_INPUT_TOKENS = (
    8192  # OpenAI embeddings per-input limit (text-embedding-3*, ada-002).
)
_MAX_INPUTS_PER_REQUEST = 2048


def _require_openai() -> tuple[ModuleType, ModuleType]:
    """Import openai and tiktoken or raise with install hint.

    Returns:
        The ``openai`` and ``tiktoken`` modules.

    Raises:
        ImportError: If openai and tiktoken are not installed.
    """
    try:
        import openai
        import tiktoken
    except ImportError as exc:
        _missing = "OpenAIEmbedder requires optional dependencies. pip install 'fastapi-semcache[embed-openai]'."
        raise ImportError(_missing) from exc
    return openai, tiktoken


def _encoding_for_model(tiktoken: ModuleType, model_name: str) -> Any:
    """Resolve a tiktoken encoding for an embedding model name.

    Args:
        tiktoken: The ``tiktoken`` module.
        model_name: OpenAI model id (e.g. ``text-embedding-3-small``).

    Returns:
        A ``tiktoken.Encoding`` instance.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


class OpenAIEmbedder(BaseEmbedder):
    """Embed text using the OpenAI ``embeddings`` API (async, batched)."""

    _model_name: str
    _dimensions: int
    _openai: ModuleType
    _api_key: str | None
    _base_url: str | None
    _send_dimensions_to_api: bool
    _client: Any | None
    _client_lock: threading.Lock
    _encoding: Any

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        dimensions: int = 1536,
        send_dimensions_to_api: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Configure the OpenAI client and declared vector width for storage.

        Args:
            model_name: Model id passed to ``embeddings.create`` (any provider id).
            dimensions: Vector width for storage and validation. When
                ``send_dimensions_to_api`` is True, this is also sent as the API
                ``dimensions`` argument for models that support it.
            send_dimensions_to_api: When True, include ``dimensions`` in the request
                body (optional output width). Set False for fixed-size models (for
                example ``text-embedding-ada-002``) that reject that parameter.
            api_key: Optional API key; defaults to ``OPENAI_API_KEY`` when omitted.
            base_url: Optional API base URL (Azure OpenAI, proxies, etc.).
        """
        openai, tiktoken = _require_openai()
        self._dimensions = BaseEmbedder.require_positive_dim(dimensions)
        self._model_name = model_name
        self._send_dimensions_to_api = send_dimensions_to_api
        self._openai = openai
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._client_lock = threading.Lock()
        self._encoding = _encoding_for_model(tiktoken, model_name)

    def _get_client(self) -> "AsyncOpenAI":
        """Return a lazily constructed ``AsyncOpenAI`` client.

        Thread-safe so concurrent ``embed`` tasks share one client.

        Returns:
            Shared async OpenAI client instance.
        """
        with self._client_lock:
            if self._client is None:
                self._client = self._openai.AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            return self._client  # type: ignore

    def _dimensions_param(self) -> dict[str, int]:
        """Build optional ``dimensions`` argument for ``embeddings.create``.

        Returns:
            Dict with a ``dimensions`` key when ``send_dimensions_to_api`` is True,
            else empty.
        """
        if self._send_dimensions_to_api:
            return {"dimensions": self._dimensions}
        return {}

    def _validate_token_counts(self, texts: list[str], offset: int) -> None:
        """Ensure each string is within the API token limit.

        Args:
            texts: Batch slice to validate.
            offset: Global index of ``texts[0]`` in the original request.

        Raises:
            ValueError: If any input exceeds ``_MAX_INPUT_TOKENS`` tokens.
        """
        for i, text in enumerate(texts):
            n_tokens = len(self._encoding.encode(text))
            if n_tokens > _MAX_INPUT_TOKENS:
                msg = (
                    f"text at index {offset + i} has {n_tokens} tokens "
                    f"(max {_MAX_INPUT_TOKENS})"
                )
                raise ValueError(msg)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for one API batch (already token-checked).

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
            **self._dimensions_param(),
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
            msg = (
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
        return f"openai:{self._model_name}:{self._dimensions}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API in chunks (token-checked per chunk).

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
            self._validate_token_counts(batch, start)
            results.extend(await self._embed_batch(batch))
        return results
