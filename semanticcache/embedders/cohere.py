"""Cohere embedding backend (official ``cohere`` async client, v2 embed API)."""

from __future__ import annotations

# cohere typing is incomplete; interactions stay runtime-checked.
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingModuleSource=false

import threading
from types import ModuleType
from typing import Any, Literal, final, override

from ..exceptions import InvalidEmbeddingDimensionException
from ._base import BaseEmbedder

_MAX_INPUTS_PER_REQUEST = 96
"""Maximum texts per Cohere embed request (documented API limit)."""

COHERE_DEFAULT_MODEL = "embed-v4.0"
"""Default Cohere embedding model used by ``get_embedder`` and ``CohereEmbedder``."""

COHERE_DEFAULT_DIMENSIONS = 1536
"""Default embedding vector width matching ``COHERE_DEFAULT_MODEL``."""

COHERE_DEFAULT_INPUT_TYPE = "search_document"
"""Default ``input_type`` for Cohere embed v3+ models."""

CohereInputType = Literal[
    "search_document",
    "search_query",
    "classification",
    "clustering",
]

CohereTruncate = Literal["NONE", "START", "END"]


def _require_cohere() -> ModuleType:
    """Import cohere or raise with install hint.

    Returns:
        The ``cohere`` module.

    Raises:
        ImportError: If cohere is not installed.
    """
    try:
        import cohere as _cohere
    except ImportError as exc:
        msg = (
            "CohereEmbedder requires optional dependencies. "
            "pip install 'fastapi-semcache[embed-cohere]'."
        )
        raise ImportError(msg) from exc
    return _cohere


@final
class CohereEmbedder(BaseEmbedder):
    """Embed text using Cohere's v2 embed API via ``cohere.AsyncClient``."""

    _model_name: str
    _dimensions: int
    _input_type: CohereInputType
    _output_dimension: int | None
    _truncate: CohereTruncate | None
    _api_key: str | None
    _base_url: str | None
    _cohere: ModuleType
    _client: Any | None
    _client_lock: threading.Lock

    def __init__(
        self,
        model_name: str = COHERE_DEFAULT_MODEL,
        *,
        dimensions: int = COHERE_DEFAULT_DIMENSIONS,
        input_type: CohereInputType = COHERE_DEFAULT_INPUT_TYPE,
        output_dimension: int | None = None,
        truncate: CohereTruncate | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Configure the Cohere async client and declared vector width for storage.

        Args:
            model_name: Cohere embed model id (for example ``embed-v4.0``,
                ``embed-english-v3.0``). Passed to ``v2.embed`` when
                ``output_dimension`` is set, else ``AsyncClient.embed``.
            dimensions: Expected embedding vector width used for storage, validation,
                and ``cache_namespace``. Must match the model output (or
                ``output_dimension`` when set).
            input_type: Cohere ``input_type`` hint (required for embed v3+). Use
                ``search_document`` when indexing cache entries and
                ``search_query`` for lookup-heavy workloads.
            output_dimension: When set, passed as ``output_dimension`` to the v2 API.
                Only supported by ``embed-v4`` and newer models (256, 512, 1024,
                1536). When used, ``dimensions`` should equal this value.
            truncate: How inputs longer than the model max token length are handled
                (``NONE``, ``START``, or ``END``). When omitted, the API default
                applies.
            api_key: Cohere API key. Defaults to the ``COHERE_API_KEY`` environment
                variable when omitted (read by ``cohere.AsyncClient``).
            base_url: Optional API base URL (enterprise or proxy deployments).
        """
        cohere_mod = _require_cohere()
        self._dimensions = BaseEmbedder.require_positive_dim(dimensions)
        self.api_key_required(api_key)
        self._model_name = model_name
        self._input_type = input_type
        self._output_dimension = output_dimension
        self._truncate = truncate
        self._api_key = api_key
        self._base_url = base_url
        self._cohere = cohere_mod
        self._client = None
        self._client_lock = threading.Lock()

    @override
    def api_key_required(self, api_key: str) -> None:
        if api_key is None:
            msg = "CohereEmbedder requires an API key"
            raise ValueError(msg)

    def _get_client(self) -> Any:
        """Return a lazily constructed ``cohere.AsyncClient``.

        Thread-safe so concurrent ``embed`` tasks share one client.

        Returns:
            Shared async Cohere client instance.
        """
        with self._client_lock:
            if self._client is None:
                self._client = self._cohere.AsyncClient(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            return self._client

    async def aclose(self) -> None:
        """Close the underlying httpx client if one was created.

        Safe to call when no client exists or when ``close`` was already invoked.
        """
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            await client._client_wrapper.httpx_client.httpx_client.aclose()

    def _build_v2_embed_kwargs(self, texts: list[str]) -> dict[str, Any]:
        """Assemble keyword arguments for ``AsyncClient.v2.embed``.

        Args:
            texts: Non-empty batch of strings for this request.

        Returns:
            Dict suitable for ``await client.v2.embed(**kwargs)``.
        """
        kwargs: dict[str, Any] = {
            "texts": texts,
            "model": self._model_name,
            "input_type": self._input_type,
            "embedding_types": ["float"],
        }
        if self._output_dimension is not None:
            kwargs["output_dimension"] = self._output_dimension
        if self._truncate is not None:
            kwargs["truncate"] = self._truncate
        return kwargs

    def _extract_float_vectors(self, response: Any) -> list[list[float]]:
        """Return float embeddings from a Cohere embed response object.

        Args:
            response: ``EmbedByTypeResponse`` (v2) or ``EmbedResponse`` (top-level).

        Returns:
            One float vector per input text, in order.

        Raises:
            RuntimeError: If the response does not contain float embeddings.
            ValueError: If the float vector count is unexpected.
        """
        response_type = getattr(response, "response_type", None)
        if response_type == "embeddings_floats":
            vectors: list[list[float]] = list(response.embeddings)
            return vectors

        embeddings_obj = getattr(response, "embeddings", None)
        if embeddings_obj is None:
            msg = "Cohere embed response missing embeddings"
            raise RuntimeError(msg)

        if hasattr(embeddings_obj, "float_"):
            floats = embeddings_obj.float_
            if floats is None:
                msg = "Cohere embed response missing float embeddings"
                raise RuntimeError(msg)
            return list(floats)

        if isinstance(embeddings_obj, list):
            return list(embeddings_obj)

        msg = "unsupported Cohere embed response shape"
        raise RuntimeError(msg)

    def _validate_vector_lengths(self, vectors: list[list[float]]) -> None:
        """Ensure each vector matches the configured storage width.

        Args:
            vectors: Embeddings returned by the API.

        Raises:
            InvalidEmbeddingDimensionException: If any vector length differs from
                ``dimensions``.
        """
        for vec in vectors:
            if len(vec) != self._dimensions:
                msg = (
                    f"expected embedding length {self._dimensions}, got {len(vec)} "
                    f"for model {self._model_name!r}"
                )
                raise InvalidEmbeddingDimensionException(msg)

    async def _embed_batch_v2(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for one v2 API batch.

        Args:
            texts: Non-empty batch of strings (at most 96).

        Returns:
            One vector per string, same order as ``texts``.

        Raises:
            RuntimeError: If the API response lacks float embeddings.
            ValueError: If the API returns the wrong number of vectors.
            InvalidEmbeddingDimensionException: If a vector length does not match
                ``dimensions``.
        """
        response = await self._get_client().v2.embed(
            **self._build_v2_embed_kwargs(texts)
        )
        vectors = self._extract_float_vectors(response)
        if len(vectors) != len(texts):
            msg = (
                f"Cohere API returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
            raise ValueError(msg)
        self._validate_vector_lengths(vectors)
        return vectors

    async def _embed_via_top_level(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts using ``AsyncClient.embed`` (SDK batches at 96).

        Used when ``output_dimension`` is unset so the convenience embed path applies.

        Args:
            texts: Non-empty list of strings.

        Returns:
            One vector per string, same order as ``texts``.

        Raises:
            RuntimeError: If the API response lacks float embeddings.
            ValueError: If the API returns the wrong number of vectors.
            InvalidEmbeddingDimensionException: If a vector length does not match
                ``dimensions``.
        """
        kwargs: dict[str, Any] = {
            "texts": texts,
            "model": self._model_name,
            "input_type": self._input_type,
            "embedding_types": ["float"],
            "batching": True,
        }
        if self._truncate is not None:
            kwargs["truncate"] = self._truncate
        response = await self._get_client().embed(**kwargs)
        vectors = self._extract_float_vectors(response)
        if len(vectors) != len(texts):
            msg = (
                f"Cohere API returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
            raise ValueError(msg)
        self._validate_vector_lengths(vectors)
        return vectors

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
        return f"cohere:{self._model_name}:{self._dimensions}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call the Cohere embed API in chunks (v2 when ``output_dimension`` is set).

        Args:
            texts: Strings to embed.

        Returns:
            Embedding vectors as nested lists, aligned with ``texts``.
        """
        if not texts:
            return []

        if self._output_dimension is None:
            return await self._embed_via_top_level(texts)

        results: list[list[float]] = []
        for start in range(0, len(texts), _MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + _MAX_INPUTS_PER_REQUEST]
            results.extend(await self._embed_batch_v2(batch))
        return results
