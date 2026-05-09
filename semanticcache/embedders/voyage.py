"""Voyage AI embedding backend (aiohttp async client + voyageai token validation)."""

from __future__ import annotations

# voyageai and aiohttp typing is incomplete; interactions stay runtime-checked.
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingModuleSource=false

import threading
from types import ModuleType
from typing import Any, final, override

from ..exceptions import InvalidEmbeddingDimensionException
from ._base import BaseEmbedder

_VOYAGE_BASE_URL = "https://api.voyageai.com/v1/embeddings"
"""Voyage AI embeddings REST endpoint."""

_MAX_INPUTS_PER_REQUEST = 1000
"""Maximum texts per Voyage API request (documented hard limit)."""

VOYAGE_DEFAULT_MODEL = "voyage-4"
"""Default Voyage embedding model used by ``get_embedder`` and ``VoyageEmbedder``."""

VOYAGE_DEFAULT_DIMENSIONS = 1024
"""Default embedding vector width matching ``VOYAGE_DEFAULT_MODEL``."""


def _require_voyageai_and_aiohttp() -> tuple[ModuleType, ModuleType]:
    """Import voyageai and aiohttp or raise with install hint.

    Returns:
        Tuple of (voyageai module, aiohttp module).

    Raises:
        ImportError: If voyageai or aiohttp are not installed.
    """
    try:
        import voyageai
        import aiohttp as _aiohttp
    except ImportError as exc:
        msg = (
            "VoyageEmbedder requires optional dependencies. "
            "pip install 'fastapi-semcache[embed-voyage]'."
        )
        raise ImportError(msg) from exc
    return voyageai, _aiohttp


@final
class VoyageEmbedder(BaseEmbedder):
    """Embed text using the Voyage AI embeddings API (aiohttp async, batched).

    Token validation uses ``voyageai.Client.tokenize`` locally (no network call)
    before each batch. HTTP requests use ``aiohttp`` with a shared session, as
    recommended by Voyage for async workloads.
    """

    _model_name: str
    _dimensions: int
    _output_dimension: int | None
    _input_type: str | None
    _api_key: str | None
    _voyageai: ModuleType
    _aiohttp: ModuleType
    _vo_client: Any
    _session: Any | None
    _session_lock: threading.Lock

    def __init__(
        self,
        model_name: str = VOYAGE_DEFAULT_MODEL,
        *,
        dimensions: int = VOYAGE_DEFAULT_DIMENSIONS,
        output_dimension: int | None = None,
        input_type: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Configure the Voyage embedder and token validator.

        Args:
            model_name: Voyage model id passed to the embeddings endpoint and to
                ``voyageai.Client.tokenize`` for per-text token validation. Defaults
                to ``voyage-3``.
            dimensions: Expected embedding vector width used for storage, validation,
                and ``cache_namespace``. Must match the model's actual output width
                (or ``output_dimension`` when set).
            output_dimension: When set, passed as ``output_dimension`` in the API
                request body. Only supported by ``voyage-4-*``, ``voyage-3-large``,
                ``voyage-3.5*``, and ``voyage-code-3``. When used, ``dimensions``
                should equal this value.
            input_type: Optional hint sent to the API. Options: ``None``, ``"query"``,
                ``"document"``. Use ``"document"`` for indexing, ``"query"`` for
                lookup to improve retrieval accuracy.
            api_key: Voyage API key. Defaults to the ``VOYAGE_API_KEY`` environment
                variable when omitted (read by ``voyageai.Client``).
        """
        voyageai_mod, aiohttp_mod = _require_voyageai_and_aiohttp()
        self._dimensions = BaseEmbedder.require_positive_dim(dimensions)
        self._model_name = model_name
        self._output_dimension = output_dimension
        self._input_type = input_type
        self._api_key = api_key
        self._voyageai = voyageai_mod
        self._aiohttp = aiohttp_mod
        self._vo_client = voyageai_mod.Client(api_key=api_key)
        self._session = None
        self._session_lock = threading.Lock()

    def _get_session(self) -> Any:
        """Return a lazily constructed ``aiohttp.ClientSession``.

        The session must be created inside a running event loop, so construction is
        deferred to the first ``embed`` call. Thread-safe via a lock so concurrent
        coroutines share one session.

        Returns:
            Shared ``aiohttp.ClientSession`` instance.
        """
        with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = self._aiohttp.ClientSession()
            return self._session

    def _validate_token_counts(self, texts: list[str], offset: int) -> None:
        """Validate each text is within the per-input context length using voyageai tokenizer.

        Uses ``voyageai.Client.tokenize`` which runs locally against the Hugging Face
        tokenizer for the configured model - no network call.

        Args:
            texts: Batch slice to validate.
            offset: Global index of ``texts[0]`` in the original request, used
                in error messages.

        Raises:
            ValueError: If any input has zero tokens or the tokenizer returns an
                unexpected number of encodings.
        """
        encodings = self._vo_client.tokenize(texts, model=self._model_name)
        if len(encodings) != len(texts):
            msg = (
                f"voyageai tokenizer returned {len(encodings)} encodings "
                f"for {len(texts)} texts"
            )
            raise ValueError(msg)

    def _build_request_body(self, texts: list[str]) -> dict[str, Any]:
        """Assemble the JSON payload for the Voyage embeddings endpoint.

        Args:
            texts: Non-empty list of strings for this batch.

        Returns:
            Dict suitable for ``json=`` in the aiohttp POST call.
        """
        body: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }
        if self._input_type is not None:
            body["input_type"] = self._input_type
        if self._output_dimension is not None:
            body["output_dimension"] = self._output_dimension
        return body

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings for one API batch (already token-checked).

        Args:
            texts: Non-empty batch of strings.

        Returns:
            One vector per string, same order as ``texts``.

        Raises:
            RuntimeError: If the API returns a non-200 status or the response
                structure is unexpected.
            ValueError: If the response vector count does not match ``texts``.
            InvalidEmbeddingDimensionException: If a returned vector length does not
                match ``dimensions``.
        """
        session = self._get_session()
        headers = {"Authorization": f"Bearer {self._api_key or ''}"}
        body = self._build_request_body(texts)

        async with session.post(
            _VOYAGE_BASE_URL,
            headers=headers,
            json=body,
        ) as response:
            if response.status != 200:
                text = await response.text()
                msg = f"Voyage API returned HTTP {response.status}: {text}"
                raise RuntimeError(msg)
            payload: dict[str, Any] = await response.json()

        rows: list[dict[str, Any]] = list(payload.get("data", []))
        if len(rows) != len(texts):
            msg = f"Voyage API returned {len(rows)} embeddings for {len(texts)} inputs"
            raise ValueError(msg)

        # rows are documented as ordered by index
        by_index: dict[int, list[float]] = {}
        for row in rows:
            idx: int = row["index"]
            if idx in by_index:
                msg = f"duplicate embedding index {idx} in Voyage API response"
                raise ValueError(msg)
            by_index[idx] = row["embedding"]

        expected = set(range(len(texts)))
        if set(by_index.keys()) != expected:
            msg = (
                f"embedding indices {sorted(by_index.keys())!r} do not match "
                f"inputs 0..{len(texts) - 1}"
            )
            raise ValueError(msg)

        out: list[list[float]] = []
        for i in range(len(texts)):
            vec = by_index[i]
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
        return f"voyage:{self._model_name}:{self._dimensions}"

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call the Voyage embeddings API in chunks (token-validated per chunk).

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
