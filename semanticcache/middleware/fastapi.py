"""FastAPI / Starlette HTTP middleware for semantic response caching."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from typing import cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import ASGIApp

from semanticcache.cache import SemanticCache
from semanticcache.types import CacheResult

_logger = logging.getLogger(__name__)


def _extract_query_from_mapping(data: dict[str, object]) -> str | None:
    """Pick a cache key string from common LLM / search JSON shapes.

    Args:
        data: Parsed JSON object body.

    Returns:
        Non-empty query text, or None if no usable field was found.
    """
    for key in ("query", "prompt", "input"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    messages = data.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
        if parts:
            return "\n".join(parts)
    return None


async def default_extract_query(request: Request, body: bytes) -> str | None:
    """Derive cache lookup text from JSON ``query`` / ``prompt`` / ``messages`` etc.

    Args:
        request: Incoming ASGI request (used for ``Content-Type``).
        body: Raw body bytes (already read from the stream).

    Returns:
        Query text for embedding, or None if the body should not be cached.
    """
    if not body.strip():
        return None
    ct = (request.headers.get("content-type") or "").lower()
    looks_json = "json" in ct or body.lstrip()[:1] in (b"{", b"[")
    if not looks_json:
        return None
    try:
        parsed: object = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return _extract_query_from_mapping(parsed)
    return None


class SemanticCacheMiddleware(BaseHTTPMiddleware):
    """Intercept requests, serve semantic cache hits, and populate the cache on miss.

    Concurrent requests with the same extracted query (and model key) coordinate so
    only one runs the downstream handler on miss; others observe a cache hit after the
    leader stores the entry (async lock per key, double-checked get).
    """

    _HEADER_CACHE: str = "X-Cache"
    _HEADER_SIMILARITY: str = "X-Cache-Similarity"
    _HEADER_SOURCE: str = "X-Cache-Source"

    _cache: SemanticCache
    _enabled: bool
    _path_prefix: str | None
    _methods: frozenset[str]
    _extract_query: Callable[[Request, bytes], Awaitable[str | None]]
    _extract_model: Callable[[Request, bytes], Awaitable[str | None]] | None
    _model_header_name: str
    _flight_lock_registry: asyncio.Lock
    _flight_locks: dict[tuple[str, str | None], asyncio.Lock]

    def __init__(
        self,
        app: ASGIApp,
        *,
        cache: SemanticCache,
        enabled: bool = True,
        path_prefix: str | None = None,
        methods: Sequence[str] | None = None,
        extract_query: Callable[
            [Request, bytes], Awaitable[str | None]
        ] = default_extract_query,
        extract_model: Callable[[Request, bytes], Awaitable[str | None]] | None = None,
        model_header_name: str = "X-Semantic-Cache-Model",
    ) -> None:
        """Attach semantic caching to a Starlette / FastAPI application.

        Args:
            app: Inner ASGI application.
            cache: Configured ``SemanticCache`` instance (shared across requests).
            enabled: When False, requests pass through unchanged.
            path_prefix: If set, only paths starting with this prefix are processed.
            methods: Uppercase HTTP methods to intercept; default is ``("POST",)``.
            extract_query: Async function mapping ``(request, body)`` to cache key
                text; return None to skip the cache for this request.
            extract_model: Optional async function for embedder routing; defaults
                to reading ``model_header_name`` and JSON ``model``.
            model_header_name: Header checked by the default model extractor.
        """
        super().__init__(app)
        self._cache = cache
        self._enabled = enabled
        self._path_prefix = path_prefix
        self._methods = frozenset(m.upper() for m in (methods or ("POST",)))
        self._extract_query = extract_query
        self._extract_model = extract_model
        self._model_header_name = model_header_name
        self._flight_lock_registry = asyncio.Lock()
        self._flight_locks = {}

    async def _get_flight_lock(
        self, query: str, model: str | None
    ) -> asyncio.Lock:
        """Return the async lock that serializes miss handling for one cache key.

        Args:
            query: Extracted cache key text.
            model: Optional model discriminator (must match ``cache.get`` / ``put``).

        Returns:
            Async lock for this ``(query, model)`` tuple.
        """
        key = (query, model)
        async with self._flight_lock_registry:
            lock = self._flight_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._flight_locks[key] = lock
            return lock

    async def _default_extract_model(self, request: Request, body: bytes) -> str | None:
        """Read model from header or JSON body.

        Args:
            request: Current request.
            body: Raw body bytes.

        Returns:
            Model name if present, else None.
        """
        h = request.headers.get(self._model_header_name)
        if isinstance(h, str) and h.strip():
            return h.strip()
        if not body.strip():
            return None
        try:
            parsed: object = json.loads(body)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            m = parsed.get("model")
            if isinstance(m, str) and m.strip():
                return m.strip()
        return None

    def _hit_headers(self, result: CacheResult) -> dict[str, str]:
        """Build response headers for a cache hit.

        Args:
            result: Successful lookup result.

        Returns:
            Header map including ``X-Cache-*`` entries.
        """
        hdrs: dict[str, str] = {
            self._HEADER_CACHE: "HIT",
            self._HEADER_SOURCE: result.source,
        }
        if result.similarity is not None:
            hdrs[self._HEADER_SIMILARITY] = f"{result.similarity:.6f}"
        return hdrs

    def _miss_headers(self) -> dict[str, str]:
        """Return headers attached to uncached or pass-through responses."""
        return {self._HEADER_CACHE: "MISS"}

    def _merge_response_headers(
        self,
        response: Response,
        extra: MutableMapping[str, str],
    ) -> None:
        """Merge ``extra`` into ``response.headers`` in place.

        Args:
            response: ASGI response whose headers are mutated.
            extra: Additional header keys and values.
        """
        for key, value in extra.items():
            response.headers[key] = value

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Run the middleware: lookup, short-circuit on hit, else forward and maybe store.

        Args:
            request: Incoming request.
            call_next: Next ASGI handler in the stack.

        Returns:
            Response from cache, or from the downstream app (possibly with MISS headers).

        Note:
            If storing a cache entry fails after the route returned a successful JSON
            body, the error is logged and that body is still returned to the client.

            For the same ``(query, model)``, concurrent misses are serialized: waiters
            re-check the cache after the leader finishes and usually avoid duplicate
            upstream work.
        """
        if not self._enabled:
            return await call_next(request)
        if request.method not in self._methods:
            return await call_next(request)
        if self._path_prefix is not None and not request.url.path.startswith(
            self._path_prefix
        ):
            return await call_next(request)

        body = await request.body()

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)

        query = await self._extract_query(request, body)
        if query is None or not str(query).strip():
            return await call_next(request)

        extract_model = self._extract_model or self._default_extract_model
        model = await extract_model(request, body)

        result = await self._cache.get(query, model=model)
        if result.is_hit and result.response is not None:
            return JSONResponse(
                content=result.response,
                headers=self._hit_headers(result),
            )

        flight = await self._get_flight_lock(query, model)
        async with flight:
            result = await self._cache.get(query, model=model)
            if result.is_hit and result.response is not None:
                return JSONResponse(
                    content=result.response,
                    headers=self._hit_headers(result),
                )

            response = await call_next(request)
            miss = self._miss_headers()

            if not (200 <= response.status_code < 300):
                self._merge_response_headers(response, miss)
                return response

            chunks: list[bytes] = []
            stream_resp = cast(StreamingResponse, response)
            # BaseHTTPMiddleware.call_next wraps the route response so the body is
            # exposed as body_iterator (typed on StreamingResponse).
            async for chunk in stream_resp.body_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode(stream_resp.charset)
                elif isinstance(chunk, memoryview):
                    chunk = bytes(chunk)
                chunks.append(chunk)
            buffered = b"".join(chunks)

            try:
                payload: object = json.loads(buffered)
            except json.JSONDecodeError:
                out = Response(
                    content=buffered,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                    background=response.background,
                )
                self._merge_response_headers(out, miss)
                return out

            merged_headers = {**dict(response.headers), **miss}
            final = Response(
                content=buffered,
                status_code=response.status_code,
                headers=merged_headers,
                media_type=response.media_type,
                background=response.background,
            )

            # Persist JSON objects on success. Do not require a JSON Content-Type: many
            # servers omit the header or use nonstandard values; the old check for the
            # substring application/json skipped put() entirely.
            if isinstance(payload, dict):
                try:
                    await self._cache.put(query, payload, model=model)
                except Exception:
                    _logger.exception(
                        "Semantic cache write failed; returning upstream response unchanged."
                    )

            return final
