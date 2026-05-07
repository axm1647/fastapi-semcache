"""FastAPI / Starlette HTTP middleware for semantic response caching."""

# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..cache import SemanticCache, resolve_cache_scope
from ..types import CacheResult
from .extractors import (
    default_extract_model,
    default_extract_query,
    default_extract_scope,
)
from .replay import (
    build_hit_headers,
    build_miss_headers,
    cache_record_from_response,
    merge_response_headers,
    response_from_cache_hit,
)

if TYPE_CHECKING:
    from ..config import CacheSettings

_logger = logging.getLogger(__name__)

_CACHE_KEY_LOG_MAX = 48


@dataclass(frozen=True, slots=True)
class ResponseValidationContext:
    """Hold response details passed to a cache store validator."""

    request: Request
    request_body: bytes
    response: Response
    payload: dict[str, object]
    model: str | None
    scope: str | None


type ResponseShapeValidator = Callable[
    [ResponseValidationContext],
    bool | Awaitable[bool],
]


def _normalize_request_path(path: str) -> str:
    """Normalize request paths so equivalent routes share one cache namespace.

    Args:
        path: Raw request path from Starlette.

    Returns:
        Normalized absolute path with duplicate trailing slash removed.
    """
    candidate = path.strip() or "/"
    if not candidate.startswith("/"):
        candidate = f"/{candidate}"
    if candidate != "/":
        candidate = candidate.rstrip("/")
    return candidate


def _compose_cache_lookup_query(
    *,
    method: str,
    normalized_path: str,
    model: str | None,
    semantic_query: str,
) -> str:
    """Build the middleware cache lookup text with route and model dimensions.

    Args:
        method: Uppercase HTTP method.
        normalized_path: Normalized request path.
        model: Optional model discriminator.
        semantic_query: Extracted semantic lookup text.

    Returns:
        A stable lookup text that scopes semantic similarity by endpoint context.
    """
    model_value = (model or "").strip() or "-"
    return (
        f"method={method}\n"
        f"path={normalized_path}\n"
        f"model={model_value}\n"
        f"query={semantic_query.strip()}"
    )


def _cache_key_snippet(query: str, max_chars: int = _CACHE_KEY_LOG_MAX) -> str:
    """Return a short, non-secret prefix of the cache key for logs.

    Args:
        query: Full cache lookup text.
        max_chars: Maximum characters before truncation.

    Returns:
        Truncated text with an ellipsis when shortened.
    """
    text = query.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _request_id_for_log(request: Request) -> str | None:
    """Best-effort request or trace id from common headers.

    Args:
        request: Current ASGI request.

    Returns:
        First non-empty id header value, capped for log safety, or None.
    """
    for name in ("X-Request-ID", "X-Correlation-ID", "X-Trace-ID"):
        raw = request.headers.get(name)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()[:128]
    return None


class SemanticCacheMiddleware:
    """Intercept requests, serve semantic cache hits, and populate the cache on miss.

    Concurrent requests with the same extracted query, model key, and scope coordinate
    so only one runs the downstream handler on miss; others observe a cache hit after
    the leader stores the entry (async lock per key, double-checked get).
    """

    _HEADER_CACHE: str = "X-Cache"
    _HEADER_CACHE_ERROR: str = "X-Cache-Error"
    _HEADER_SIMILARITY: str = "X-Cache-Similarity"
    _HEADER_SOURCE: str = "X-Cache-Source"
    _HEADER_CIRCUIT: str = "X-Cache-Circuit"
    _CACHE_RECORD_MARKER: str = "__semanticcache_record_v1__"

    _cache: SemanticCache
    _enabled: bool
    _path_prefix: str | None
    _methods: frozenset[str]
    _extract_query: Callable[[Request, bytes], Awaitable[str | None]]
    _extract_model: Callable[[Request, bytes], Awaitable[str | None]] | None
    _model_header_name: str
    _extract_scope: Callable[[Request, bytes], Awaitable[str | None]] | None
    _scope_header_name: str
    _validate_response: ResponseShapeValidator | None
    _scope_settings: CacheSettings
    _require_cache_scope: bool
    _flight_lock_registry: asyncio.Lock
    _flight_locks: OrderedDict[tuple[str, str | None, str], asyncio.Lock]
    _flight_lock_max_entries: int
    _circuit_breaker_enabled: bool
    _circuit_breaker_limit: int
    _circuit_breaker_open_seconds: float
    _cache_authorized_requests: bool
    _circuit_lock: asyncio.Lock
    _consecutive_429_count: int
    _circuit_open_until: float | None
    app: ASGIApp

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
        extract_scope: Callable[[Request, bytes], Awaitable[str | None]] | None = None,
        scope_header_name: str = "X-Semantic-Cache-Scope",
        validate_response: ResponseShapeValidator | None = None,
        cache_settings: CacheSettings | None = None,
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
            extract_scope: Optional async tenant or namespace extractor. When
                tenant scope is required, the default reads ``scope_header_name`` and
                JSON ``cache_scope`` / ``tenant_id`` (including integer ``tenant_id``).
            scope_header_name: Header checked by the default scope extractor.
            validate_response: Optional sync or async callback that receives a
                ``ResponseValidationContext`` before a successful JSON object is stored.
                Return False to skip storing malformed or route-mismatched payloads.
            cache_settings: Optional settings override; defaults to
                ``get_cache_settings()`` (429 circuit breaker and flight-lock cap).
                When ``cache`` exposes a ``settings`` attribute (as ``SemanticCache``
                does), ``require_cache_scope`` and the middleware scope gate use it so
                they stay aligned with ``SemanticCache``; otherwise ``cache_settings``
                applies. This source also controls whether requests that include an
                ``Authorization`` header are cacheable.
        """
        from ..config import get_cache_settings

        self.app = app
        self._cache = cache
        self._enabled = enabled
        self._path_prefix = path_prefix
        self._methods = frozenset(m.upper() for m in (methods or ("POST",)))
        self._extract_query = extract_query
        self._extract_model = extract_model
        self._model_header_name = model_header_name
        self._extract_scope = extract_scope
        self._scope_header_name = scope_header_name
        self._validate_response = validate_response
        self._flight_lock_registry = asyncio.Lock()
        resolved = (
            cache_settings if cache_settings is not None else get_cache_settings()
        )
        self._cache_settings = resolved
        cache_settings_obj = getattr(cache, "settings", None)
        if cache_settings_obj is not None:
            self._scope_settings = cache_settings_obj
            self._require_cache_scope = cache_settings_obj.require_cache_scope
        else:
            self._scope_settings = resolved
            self._require_cache_scope = resolved.require_cache_scope
        self._flight_locks = OrderedDict()
        self._flight_lock_max_entries = max(
            1, resolved.middleware_flight_lock_max_entries
        )
        self._circuit_breaker_enabled = resolved.circuit_breaker_429_enabled
        self._circuit_breaker_limit = resolved.circuit_breaker_429_consecutive_limit
        self._circuit_breaker_open_seconds = resolved.circuit_breaker_429_open_seconds
        self._cache_authorized_requests = resolved.cache_authorized_requests
        self._circuit_lock = asyncio.Lock()
        self._consecutive_429_count = 0
        self._circuit_open_until = None

    def _evict_unused_flight_locks(self) -> None:
        """Evict least-recently-used unlocked flight locks when over capacity.

        This method expects ``self._flight_lock_registry`` to already be held.
        Locked entries are preserved so active in-flight request coordination is
        never broken. If every entry is currently locked, the registry can
        temporarily remain above the configured cap until one becomes idle.
        """
        while len(self._flight_locks) > self._flight_lock_max_entries:
            removed_any = False
            for key, lock in list(self._flight_locks.items()):
                if lock.locked():
                    continue
                self._flight_locks.pop(key, None)
                removed_any = True
                break
            if not removed_any:
                return

    async def _get_flight_lock(
        self, query: str, model: str | None, scope_storage: str
    ) -> asyncio.Lock:
        """Return the async lock that serializes miss handling for one cache key.

        Args:
            query: Extracted cache key text.
            model: Optional model discriminator (must match ``cache.get`` / ``put``).
            scope_storage: Resolved scope string passed to ``SemanticCache`` (may be
                empty when ``require_cache_scope`` is False).

        Returns:
            Async lock for this ``(query, model, scope)`` tuple.
        """
        key = (query, model, scope_storage)
        async with self._flight_lock_registry:
            lock = self._flight_locks.get(key)
            if lock is not None:
                self._flight_locks.move_to_end(key)
                return lock
            lock = asyncio.Lock()
            self._flight_locks[key] = lock
            self._evict_unused_flight_locks()
            return lock

    async def _upstream_blocked_by_circuit(self) -> bool:
        """Return True when the 429 circuit is open and upstream must not be called.

        Expired open windows are cleared while holding the circuit lock.

        Returns:
            True if ``call_next`` must be skipped and only cache hits apply.
        """
        if not self._circuit_breaker_enabled:
            return False
        async with self._circuit_lock:
            if self._circuit_open_until is None:
                return False
            now = time.monotonic()
            if now >= self._circuit_open_until:
                self._circuit_open_until = None
                return False
            return True

    async def _record_upstream_status_for_circuit(self, status_code: int) -> None:
        """Count consecutive HTTP 429 responses and open the circuit when tripped."""
        if not self._circuit_breaker_enabled:
            return
        async with self._circuit_lock:
            if status_code == 429:
                self._consecutive_429_count += 1
                if self._consecutive_429_count >= self._circuit_breaker_limit:
                    self._circuit_open_until = (
                        time.monotonic() + self._circuit_breaker_open_seconds
                    )
                    self._consecutive_429_count = 0
                    _logger.warning(
                        "Semantic cache 429 circuit breaker opened for %.2f seconds "
                        "after %i consecutive 429 response(s) from upstream",
                        self._circuit_breaker_open_seconds,
                        self._circuit_breaker_limit,
                    )
            else:
                self._consecutive_429_count = 0

    async def _default_extract_model(self, request: Request, body: bytes) -> str | None:
        """Read model from header or JSON body.

        Args:
            request: Current request.
            body: Raw body bytes.

        Returns:
            Model name if present, else None.
        """
        return await default_extract_model(
            request,
            body,
            model_header_name=self._model_header_name,
        )

    async def _default_extract_scope(self, request: Request, body: bytes) -> str | None:
        """Read tenant or namespace scope from header or JSON body.

        Args:
            request: Current request.
            body: Raw body bytes.

        Returns:
            Non-empty scope string when present, else None.
        """
        return await default_extract_scope(
            request,
            body,
            scope_header_name=self._scope_header_name,
        )

    def _hit_headers(self, result: CacheResult) -> dict[str, str]:
        """Build response headers for a cache hit.

        Args:
            result: Successful lookup result.

        Returns:
            Header map including ``X-Cache-*`` entries.
        """
        return build_hit_headers(
            result=result,
            cache_header_name=self._HEADER_CACHE,
            source_header_name=self._HEADER_SOURCE,
            similarity_header_name=self._HEADER_SIMILARITY,
        )

    def _miss_headers(self, *, cache_read_error: bool = False) -> dict[str, str]:
        """Return headers attached to uncached or pass-through responses.

        Args:
            cache_read_error: When True, add ``X-Cache-Error: 1`` (read path failed).

        Returns:
            Header map with ``X-Cache: MISS`` and optional error marker.
        """
        return build_miss_headers(
            cache_header_name=self._HEADER_CACHE,
            cache_error_header_name=self._HEADER_CACHE_ERROR,
            cache_read_error=cache_read_error,
        )

    def _log_cache_get_failure(
        self,
        request: Request,
        *,
        query: str,
        model: str | None,
        scope: str | None = None,
        phase: str,
        exc: Exception,
    ) -> None:
        """Emit a single structured warning when ``cache.get`` fails.

        Args:
            request: Current request (path and optional request id only).
            query: Cache key text; only a snippet is logged.
            model: Optional model key; truncated in logs.
            scope: Optional tenant scope; truncated in logs.
            phase: ``preflight`` or ``double_check`` for disambiguation.
            exc: The exception raised by the cache layer.
        """
        rid = _request_id_for_log(request)
        snippet = _cache_key_snippet(query)
        model_s = (model or "").strip()[:64] or "-"
        scope_s = (scope or "").strip()[:64] or "-"
        _logger.warning(
            "Semantic cache read failed (%s): route=%s request_id=%s "
            "cache_key_snippet=%r model=%s scope=%s error=%s: %s",
            phase,
            request.url.path,
            rid if rid is not None else "-",
            snippet,
            model_s,
            scope_s,
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    def _log_extraction_failure(
        self,
        request: Request,
        *,
        phase: str,
        exc: Exception,
    ) -> None:
        """Emit a diagnostic warning when an extractor raises.

        Args:
            request: Current request (path and optional request id only).
            phase: ``extract_query``, ``extract_model``, or ``extract_scope`` for log
                filtering.
            exc: The exception raised by the extractor.
        """
        rid = _request_id_for_log(request)
        _logger.warning(
            "Semantic cache extraction failed (%s): route=%s request_id=%s "
            "error=%s: %s",
            phase,
            request.url.path,
            rid if rid is not None else "-",
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    async def _cache_get_fail_open(
        self,
        request: Request,
        query: str,
        model: str | None,
        *,
        scope: str | None,
        storage_scope_key: str,
        phase: str,
    ) -> tuple[CacheResult, bool]:
        """Run ``cache.get`` and map failures to a miss without raising.

        Args:
            request: Current request (for logging context).
            query: Cache lookup text.
            model: Optional embedder routing key.
            scope: Optional tenant or namespace for logs (raw extractor output).
            storage_scope_key: Key already resolved via ``resolve_cache_scope`` for
                this request (passed to ``SemanticCache.get`` as ``storage_scope_key``).
            phase: Log label for this read (preflight vs double_check).

        Returns:
            ``(result, cache_read_error)`` where ``cache_read_error`` is True if
            ``get`` raised and the result is a synthetic miss.
        """
        try:
            return (
                await self._cache.get(
                    query,
                    model=model,
                    storage_scope_key=storage_scope_key,
                ),
                False,
            )
        except Exception as exc:
            self._log_cache_get_failure(
                request,
                query=query,
                model=model,
                scope=scope,
                phase=phase,
                exc=exc,
            )
            return (CacheResult(is_hit=False), True)

    def _merge_response_headers(
        self,
        response: Response,
        extra: Mapping[str, str],
    ) -> None:
        """Merge ``extra`` into ``response.headers`` in place.

        Args:
            response: ASGI response whose headers are mutated.
            extra: Additional header keys and values.
        """
        merge_response_headers(response, extra)

    def _cache_record_from_response(
        self,
        *,
        payload: dict[str, object],
        response: Response,
    ) -> dict[str, object]:
        """Build a cache record with payload and response replay metadata.

        Args:
            payload: Parsed JSON object body from the upstream response.
            response: Upstream response to mirror on future cache hits.

        Returns:
            Cache record dictionary with JSON body plus replay metadata.
        """
        return cache_record_from_response(
            payload=payload,
            response=response,
            cache_record_marker=self._CACHE_RECORD_MARKER,
        )

    def _response_allows_cache_store(self, response: Response) -> bool:
        """Return True when upstream response headers permit cache storage.

        Args:
            response: Upstream response candidate for cache persistence.

        Returns:
            False when ``Cache-Control`` has ``no-store`` or ``private``, or when
            ``Set-Cookie`` is present.
        """
        if response.headers.get("set-cookie") is not None:
            return False
        cache_control = response.headers.get("cache-control")
        if cache_control is None:
            return True
        directives = {
            part.strip().lower() for part in cache_control.split(",") if part.strip()
        }
        return "no-store" not in directives and "private" not in directives

    async def _response_shape_allows_cache_store(
        self,
        context: ResponseValidationContext,
    ) -> bool:
        """Return True when the optional response validator accepts a payload.

        Args:
            context: Response details and parsed JSON payload to validate.

        Returns:
            True when no validator is configured, or when the validator accepts the
            payload. False means the response is returned but not stored.
        """
        if self._validate_response is None:
            return True
        try:
            result = self._validate_response(context)
            if isawaitable(result):
                result = await result
        except Exception as exc:
            rid = _request_id_for_log(context.request)
            _logger.warning(
                "Semantic cache response validation failed: route=%s request_id=%s "
                "model=%s scope=%s error=%s: %s",
                context.request.url.path,
                rid if rid is not None else "-",
                (context.model or "").strip()[:64] or "-",
                (context.scope or "").strip()[:64] or "-",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return False
        if result:
            return True
        _logger.debug(
            "Semantic cache response validation rejected store: route=%s model=%s "
            "scope=%s",
            context.request.url.path,
            (context.model or "").strip()[:64] or "-",
            (context.scope or "").strip()[:64] or "-",
        )
        return False

    def _response_from_cache_hit(
        self,
        *,
        result: CacheResult,
    ) -> Response | None:
        """Convert a cache hit result to the HTTP response sent to clients.

        Args:
            result: Cache lookup output with payload and similarity metadata.

        Returns:
            Response with original status and headers when metadata exists.
            Returns None when hit payload is not replayable.
        """
        return response_from_cache_hit(
            result=result,
            cache_record_marker=self._CACHE_RECORD_MARKER,
            cache_header_name=self._HEADER_CACHE,
            source_header_name=self._HEADER_SOURCE,
            similarity_header_name=self._HEADER_SIMILARITY,
        )

    async def _read_body(self, receive: Receive) -> bytes:
        """Read and buffer the incoming request body from ASGI ``receive``.

        Args:
            receive: ASGI receive callable for the current request.

        Returns:
            Full request body bytes.
        """
        chunks: list[bytes] = []
        while True:
            message = await receive()
            msg_type = message["type"]
            if msg_type == "http.disconnect":
                break
            if msg_type != "http.request":
                continue
            chunk = message.get("body", b"")
            if isinstance(chunk, bytes) and chunk:
                chunks.append(chunk)
            if not bool(message.get("more_body", False)):
                break
        return b"".join(chunks)

    async def _call_downstream(self, scope: Scope, body: bytes) -> Response:
        """Invoke downstream ASGI app and buffer its response.

        Args:
            scope: Current request ASGI scope.
            body: Full buffered request body.

        Returns:
            Buffered Starlette ``Response`` built from downstream ASGI messages.
        """
        status_code = 500
        response_headers: dict[str, str] = {}
        response_body: list[bytes] = []
        body_sent = False

        async def replay_receive() -> Message:
            nonlocal body_sent
            if body_sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def capture_send(message: Message) -> None:
            nonlocal status_code, response_headers
            msg_type = message["type"]
            if msg_type == "http.response.start":
                status_code = int(message["status"])
                raw_headers = message.get("headers", [])
                parsed_headers: dict[str, str] = {}
                for key, value in raw_headers:
                    if isinstance(key, bytes) and isinstance(value, bytes):
                        parsed_headers[key.decode("latin-1")] = value.decode("latin-1")
                response_headers = parsed_headers
                return
            if msg_type == "http.response.body":
                chunk = message.get("body", b"")
                if isinstance(chunk, bytes) and chunk:
                    response_body.append(chunk)
                return

        await self.app(scope, replay_receive, capture_send)
        return Response(
            content=b"".join(response_body),
            status_code=status_code,
            headers=response_headers,
        )

    async def _send_response(
        self,
        response: Response,
        scope: Scope,
        send: Send,
    ) -> None:
        """Emit a Starlette ``Response`` over ASGI ``send``.

        Args:
            response: Response object to emit.
            scope: Current request ASGI scope.
            send: ASGI send callable.
        """

        async def _unused_receive() -> Message:
            return {"type": "http.disconnect"}

        await response(scope, _unused_receive, send)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Run the middleware: lookup, short-circuit on hit, else forward and maybe store.

        Args:
            scope: Incoming ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.

        Note:
            If ``cache.get`` raises, the failure is logged and the request continues as
            a miss (downstream handler runs). Responses then include ``X-Cache: MISS``
            and ``X-Cache-Error: 1`` when any read in the preflight or double-check
            step failed.

            If ``extract_query``, ``extract_model``, or ``extract_scope`` raises, the
            failure is logged and the request bypasses the cache entirely (same as a
            transparent pass-through).

            If storing a cache entry fails after the route returned a successful JSON
            body, the error is logged and that body is still returned to the client.

            For the same ``(query, model, scope)``, concurrent misses are serialized:
            waiters re-check the cache after the leader finishes and usually avoid
            duplicate upstream work.

            When ``circuit_breaker_429_enabled`` is set on ``CacheSettings`` (or via
            ``cache_settings``), enough consecutive upstream HTTP 429 responses open
            a cooldown window where ``call_next`` is skipped and only cache hits are
            returned; cache misses receive ``503`` with ``X-Cache-Circuit: OPEN``.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        if not self._enabled:
            await self.app(scope, receive, send)
            return
        if request.method not in self._methods:
            await self.app(scope, receive, send)
            return
        if self._path_prefix is not None and not request.url.path.startswith(
            self._path_prefix
        ):
            await self.app(scope, receive, send)
            return
        if (
            not self._cache_authorized_requests
            and request.headers.get("authorization") is not None
        ):
            await self.app(scope, receive, send)
            return
        body = await self._read_body(receive)
        body_replayed = False

        async def _request_receive() -> Message:
            nonlocal body_replayed
            if body_replayed:
                return {"type": "http.request", "body": b"", "more_body": False}
            body_replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive=_request_receive)

        try:
            semantic_query = await self._extract_query(request, body)
        except Exception as exc:
            self._log_extraction_failure(request, phase="extract_query", exc=exc)
            passthrough = await self._call_downstream(scope, body)
            await self._send_response(passthrough, scope, send)
            return

        if semantic_query is None or not str(semantic_query).strip():
            passthrough = await self._call_downstream(scope, body)
            await self._send_response(passthrough, scope, send)
            return

        extract_model = self._extract_model or self._default_extract_model
        try:
            model = await extract_model(request, body)
        except Exception as exc:
            self._log_extraction_failure(request, phase="extract_model", exc=exc)
            passthrough = await self._call_downstream(scope, body)
            await self._send_response(passthrough, scope, send)
            return
        normalized_path = _normalize_request_path(request.url.path)
        query = _compose_cache_lookup_query(
            method=request.method.upper(),
            normalized_path=normalized_path,
            model=model,
            semantic_query=semantic_query,
        )

        raw_scope: str | None = None
        if self._require_cache_scope:
            scope_extractor = self._extract_scope or self._default_extract_scope
            try:
                raw_scope = await scope_extractor(request, body)
            except Exception as exc:
                self._log_extraction_failure(request, phase="extract_scope", exc=exc)
                passthrough = await self._call_downstream(scope, body)
                await self._send_response(passthrough, scope, send)
                return
        elif self._extract_scope is not None:
            try:
                raw_scope = await self._extract_scope(request, body)
            except Exception as exc:
                self._log_extraction_failure(request, phase="extract_scope", exc=exc)
                passthrough = await self._call_downstream(scope, body)
                await self._send_response(passthrough, scope, send)
                return

        scope_storage = resolve_cache_scope(raw_scope, settings=self._scope_settings)
        if scope_storage is None:
            passthrough = await self._call_downstream(scope, body)
            await self._send_response(passthrough, scope, send)
            return

        result, cache_read_error = await self._cache_get_fail_open(
            request,
            query,
            model,
            scope=raw_scope,
            storage_scope_key=scope_storage,
            phase="preflight",
        )
        if result.is_hit:
            cached_response = self._response_from_cache_hit(result=result)
            if cached_response is not None:
                await self._send_response(
                    cached_response,
                    scope,
                    send,
                )
                return

        flight = await self._get_flight_lock(query, model, scope_storage)
        async with flight:
            result, inner_err = await self._cache_get_fail_open(
                request,
                query,
                model,
                scope=raw_scope,
                storage_scope_key=scope_storage,
                phase="double_check",
            )
            cache_read_error = cache_read_error or inner_err
            if result.is_hit:
                cached_response = self._response_from_cache_hit(result=result)
                if cached_response is not None:
                    await self._send_response(
                        cached_response,
                        scope,
                        send,
                    )
                    return

            if await self._upstream_blocked_by_circuit():
                miss = self._miss_headers(cache_read_error=cache_read_error)
                circuit_hdrs = {
                    **miss,
                    self._HEADER_CIRCUIT: "OPEN",
                }
                await self._send_response(
                    JSONResponse(
                        status_code=503,
                        content={
                            "detail": (
                                "Upstream is temporarily not contacted after repeated "
                                "HTTP 429 responses; only cache hits are served until the "
                                "cooldown elapses."
                            )
                        },
                        headers=circuit_hdrs,
                    ),
                    scope,
                    send,
                )
                return

            response = await self._call_downstream(scope, body)
            await self._record_upstream_status_for_circuit(response.status_code)
            miss = self._miss_headers(cache_read_error=cache_read_error)

            if not (200 <= response.status_code < 300):
                self._merge_response_headers(response, miss)
                await self._send_response(response, scope, send)
                return

            buffered = bytes(response.body)

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
                await self._send_response(out, scope, send)
                return

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
            if (
                isinstance(payload, dict)
                and self._response_allows_cache_store(response)
                and await self._response_shape_allows_cache_store(
                    ResponseValidationContext(
                        request=request,
                        request_body=body,
                        response=response,
                        payload=payload,
                        model=model,
                        scope=raw_scope,
                    )
                )
            ):
                try:
                    cache_record = self._cache_record_from_response(
                        payload=payload,
                        response=response,
                    )
                    await self._cache.put(
                        query,
                        cache_record,
                        model=model,
                        storage_scope_key=scope_storage,
                    )
                except Exception:
                    _logger.exception(
                        "Semantic cache write failed; returning upstream response unchanged."
                    )

            await self._send_response(final, scope, send)
            return
