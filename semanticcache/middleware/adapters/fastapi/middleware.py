"""FastAPI / Starlette HTTP middleware for semantic response caching."""

# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ....cache import SemanticCache
from ....types import CacheResult
from .asgi_io import (
    DEFAULT_MAX_BODY_BYTES,
    call_downstream,
    read_body,
    send_response,
)
from .cache_ops import (
    cache_get_fail_open,
    response_allows_cache_store,
    response_shape_allows_cache_store,
)
from .flow import (
    extract_lookup_context_or_passthrough,
    maybe_store_cache_entry,
    prepare_response_for_client,
    send_cache_hit_if_available,
    send_circuit_open_response,
    send_passthrough,
    stream_tee_and_store,
)
from .logging_utils import (
    log_cache_get_failure,
    log_extraction_failure,
    log_response_validation_failure,
    log_response_validation_rejected,
)
from .types import ResponseShapeValidator, ResponseValidationContext
from ...core.extractors import (
    default_extract_model,
    default_extract_query,
    default_extract_scope_from_request_context,
)
from ...core.coordination import MiddlewareCoordination
from ...core.replay import (
    build_hit_headers,
    build_miss_headers,
    cache_record_from_response,
    merge_response_headers,
    response_from_cache_hit,
)

if TYPE_CHECKING:
    from ....config import CacheSettings

_logger = logging.getLogger(__name__)


def _put_accepts_query_embedding_kwarg(put_method: Callable[..., object]) -> bool:
    """Return whether ``put`` accepts ``query_embedding`` (named or via ``**kwargs``).

    Args:
        put_method: Bound or unbound ``put`` callable from a cache implementation.

    Returns:
        True when the signature includes ``query_embedding`` or a variadic keyword
        collector (``**kwargs``). False when the signature cannot be inspected.
    """
    try:
        sig = inspect.signature(put_method)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == "query_embedding":
            return True
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


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
    _coordination: MiddlewareCoordination
    _cache_authorized_requests: bool
    _cache_put_accepts_query_embedding: bool
    _max_request_body_bytes: int | None
    _max_response_body_bytes: int | None
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
        max_request_body_bytes: int | None = DEFAULT_MAX_BODY_BYTES,
        max_response_body_bytes: int | None = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        """Attach semantic caching to a Starlette / FastAPI application.

        Args:
            app: Inner ASGI application.
            cache: Configured ``SemanticCache`` instance (shared across requests).
                Duck-typed caches are supported; ``put`` should accept
                ``query_embedding`` as an optional keyword when miss lookups produce
                an embedding to reuse (see ``SemanticCache.put``). At initialization the
                middleware inspects ``cache.put`` and omits that argument when the
                signature does not accept it.
            enabled: When False, requests pass through unchanged.
            path_prefix: If set, only paths starting with this prefix are processed.
            methods: Uppercase HTTP methods to intercept; default is ``("POST",)``.
            extract_query: Async function mapping ``(request, body)`` to cache key
                text; return None to skip the cache for this request.
            extract_model: Optional async function for embedder routing; defaults
                to reading ``model_header_name`` and JSON ``model``.
            model_header_name: Header checked by the default model extractor.
            extract_scope: Optional async tenant or namespace extractor. When
                omitted and scope is required, the middleware uses
                ``default_extract_scope_from_request_context``, which reads
                ``scope_header_name`` and JSON ``cache_scope`` / ``tenant_id``
                (including integer ``tenant_id``). That default trusts client
                headers and body; multi-tenant production apps should pass an
                extractor that resolves scope from server-side identity (see
                ``trusted_extract_scope_from_server_side`` in
                ``semanticcache.middleware.core.extractors``).
            scope_header_name: Header checked when using the request-context default
                scope extractor.
            validate_response: Optional sync or async callback that receives a
                ``ResponseValidationContext`` before a successful JSON object is stored.
                Return False to skip storing malformed or route-mismatched payloads.
            cache_settings: Optional settings override; defaults to
                ``get_cache_settings()`` (429 circuit breaker, flight-lock cap, and
                ``response_mode`` when the cache does not supply its own settings).
                When ``cache`` exposes a ``settings`` attribute (as ``SemanticCache``
                does), ``require_cache_scope``, ``response_mode``, and the middleware
                scope gate use it so they stay aligned with ``SemanticCache``;
                otherwise ``cache_settings`` applies. This source also controls whether
                requests that include an ``Authorization`` header are cacheable. When
                both ``cache_settings`` and ``cache.settings`` are provided and
                disagree on user-facing flags (``require_cache_scope`` or
                ``cache_authorized_requests``), a warning is logged so a likely
                misconfiguration is visible at startup.
            max_request_body_bytes: Maximum size of the buffered request body (default
                ``DEFAULT_MAX_BODY_BYTES``, 10 MiB). When exceeded, the client receives
                HTTP 413. Set to ``None`` to disable the limit (not recommended in
                production).
            max_response_body_bytes: Maximum size of the buffered downstream response
                body (default ``DEFAULT_MAX_BODY_BYTES``, 10 MiB). When exceeded, the
                client receives HTTP 502. Set to ``None`` to disable the limit.
        """
        from ....config import get_cache_settings

        self.app = app
        self._cache = cache
        put_method = getattr(cache, "put", None)
        self._cache_put_accepts_query_embedding = callable(
            put_method
        ) and _put_accepts_query_embedding_kwarg(put_method)
        self._enabled = enabled
        self._path_prefix = path_prefix
        self._methods = frozenset(m.upper() for m in (methods or ("POST",)))
        self._extract_query = extract_query
        self._extract_model = extract_model
        self._model_header_name = model_header_name
        self._extract_scope = extract_scope
        self._scope_header_name = scope_header_name
        self._validate_response = validate_response
        resolved = (
            cache_settings if cache_settings is not None else get_cache_settings()
        )
        self._cache_settings = resolved
        cache_settings_obj = getattr(cache, "settings", None)
        if cache_settings is not None and cache_settings_obj is not None:
            self._warn_on_settings_mismatch(
                middleware_settings=cache_settings,
                cache_settings=cache_settings_obj,
            )
        if cache_settings_obj is not None:
            self._scope_settings = cache_settings_obj
            self._require_cache_scope = cache_settings_obj.require_cache_scope
        else:
            self._scope_settings = resolved
            self._require_cache_scope = resolved.require_cache_scope
        self._coordination = MiddlewareCoordination(
            flight_lock_max_entries=resolved.middleware_flight_lock_max_entries,
            circuit_breaker_enabled=resolved.circuit_breaker_429_enabled,
            circuit_breaker_limit=resolved.circuit_breaker_429_consecutive_limit,
            circuit_breaker_open_seconds=resolved.circuit_breaker_429_open_seconds,
        )
        self._cache_authorized_requests = resolved.cache_authorized_requests
        self._max_request_body_bytes = max_request_body_bytes
        self._max_response_body_bytes = max_response_body_bytes

    @staticmethod
    def _warn_on_settings_mismatch(
        *,
        middleware_settings: CacheSettings,
        cache_settings: CacheSettings,
    ) -> None:
        """Log a warning when split ``CacheSettings`` sources disagree.

        ``SemanticCacheMiddleware`` reads ``require_cache_scope`` from
        ``cache.settings`` and ``cache_authorized_requests`` (plus the circuit
        breaker and flight-lock options) from the ``cache_settings`` kwarg.
        When both sources are supplied and disagree on user-facing flags, the
        split is almost always a configuration mistake (for example, the
        middleware permits caching authorized requests but the cache enforces
        scope, or vice versa). We warn rather than raise so applications that
        intentionally split these can still start.

        Args:
            middleware_settings: ``CacheSettings`` passed via the middleware
                ``cache_settings`` kwarg.
            cache_settings: ``CacheSettings`` exposed on the ``SemanticCache``
                instance (``cache.settings``).
        """
        mismatched_fields = ("require_cache_scope", "cache_authorized_requests")
        for field_name in mismatched_fields:
            middleware_value = getattr(middleware_settings, field_name)
            cache_value = getattr(cache_settings, field_name)
            if middleware_value != cache_value:
                _logger.warning(
                    "SemanticCacheMiddleware settings mismatch: "
                    "cache_settings.%s=%r (middleware kwarg) differs from "
                    "cache.settings.%s=%r. cache.settings is used for the "
                    "scope gate, while cache_settings controls the circuit "
                    "breaker, flight lock, and Authorization gating. "
                    "Confirm the split is intentional or align the two sources.",
                    field_name,
                    middleware_value,
                    field_name,
                    cache_value,
                )

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

    async def _default_extract_scope_from_request_context(
        self, request: Request, body: bytes
    ) -> str | None:
        """Read scope via ``default_extract_scope_from_request_context``.

        Args:
            request: Current request.
            body: Raw body bytes.

        Returns:
            Non-empty scope string when present, else None.
        """
        return await default_extract_scope_from_request_context(
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
        log_cache_get_failure(
            request=request,
            query=query,
            model=model,
            scope=scope,
            phase=phase,
            exc=exc,
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
        log_extraction_failure(
            request=request,
            phase=phase,
            exc=exc,
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
        return await cache_get_fail_open(
            cache_get=lambda q, m, storage: self._cache.get(
                q,
                model=m,
                storage_scope_key=storage,
            ),
            query=query,
            model=model,
            scope=scope,
            storage_scope_key=storage_scope_key,
            on_failure=lambda q, m, scp, ph, exc: self._log_cache_get_failure(
                request,
                query=q,
                model=m,
                scope=scp,
                phase=ph,
                exc=exc,
            ),
            phase=phase,
        )

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
        return response_allows_cache_store(response)

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
        return await response_shape_allows_cache_store(
            context=context,
            validate_response=self._validate_response,
            on_validation_failure=lambda ctx, exc: log_response_validation_failure(
                request=ctx.request,
                model=ctx.model,
                scope=ctx.scope,
                exc=exc,
            ),
            on_validation_rejected=lambda ctx: log_response_validation_rejected(
                request=ctx.request,
                model=ctx.model,
                scope=ctx.scope,
            ),
        )

    def _response_from_cache_hit(
        self,
        *,
        result: CacheResult,
    ) -> Response | None:
        """Convert a cache hit result to the HTTP response sent to clients.

        Args:
            result: Cache lookup output with payload and similarity metadata.

        Returns:
            Response with original status and headers when the stored record uses
            the replay envelope. Returns None when the hit payload is not
            replayable (for example missing the replay envelope).
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
        return await read_body(
            receive, max_body_bytes=self._max_request_body_bytes
        )

    async def _call_downstream(self, scope: Scope, body: bytes) -> Response:
        """Invoke downstream ASGI app and buffer its response.

        Args:
            scope: Current request ASGI scope.
            body: Full buffered request body.

        Returns:
            Buffered Starlette ``Response`` built from downstream ASGI messages.
        """
        return await call_downstream(
            self.app,
            scope,
            body,
            max_body_bytes=self._max_response_body_bytes,
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
        await send_response(response, scope, send)

    async def _cache_put_with_optional_embedding(
        self,
        query: str,
        record: dict[str, object],
        model: str | None,
        storage_scope_key: str,
        query_embedding: list[float] | None,
    ) -> None:
        """Store a cache record, passing ``query_embedding`` when supported.

        Args:
            query: Lookup query string.
            record: Cache payload and replay metadata.
            model: Optional model discriminator.
            storage_scope_key: Resolved scope key for storage.
            query_embedding: Optional precomputed embedding from a miss lookup.
        """
        if self._cache_put_accepts_query_embedding:
            await self._cache.put(
                query,
                record,
                model=model,
                storage_scope_key=storage_scope_key,
                query_embedding=query_embedding,
            )
            return
        await self._cache.put(
            query,
            record,
            model=model,
            storage_scope_key=storage_scope_key,
        )

    def _shape_validator(
        self,
    ) -> "Callable[[Request, bytes, Response, dict[str, object], str | None, str | None], Awaitable[bool]]":
        """Return an async callable that validates response shape for cache storage.

        Wraps ``_response_shape_allows_cache_store`` so both the buffered and tee
        paths share a single construction point for ``ResponseValidationContext``.
        If ``ResponseValidationContext`` gains new fields in the future, only this
        method needs updating.

        Returns:
            Async function with the signature expected by ``maybe_store_cache_entry``
            and ``stream_tee_and_store``.
        """

        async def _validate(
            req: Request,
            req_body: bytes,
            resp: Response,
            pld: dict[str, object],
            mdl: str | None,
            scp: str | None,
        ) -> bool:
            return await self._response_shape_allows_cache_store(
                ResponseValidationContext(
                    request=req,
                    request_body=req_body,
                    response=resp,
                    payload=pld,
                    model=mdl,
                    scope=scp,
                )
            )

        return _validate

    def _record_builder(
        self,
    ) -> "Callable[[dict[str, object], Response], dict[str, object]]":
        """Return a callable that builds a cache record from a payload and response.

        Returns:
            Function with the signature expected by ``maybe_store_cache_entry``
            and ``stream_tee_and_store``.
        """
        return lambda pld, resp: self._cache_record_from_response(
            payload=pld, response=resp
        )

    def _put_callback(
        self,
    ) -> "Callable[[str, dict[str, object], str | None, str, list[float] | None], Awaitable[None]]":
        """Return an async callable that persists a cache record.

        Returns:
            Function with the signature expected by ``maybe_store_cache_entry``
            and ``stream_tee_and_store``.
        """
        return lambda q, record, mdl, storage, embedding: self._cache_put_with_optional_embedding(
            q, record, mdl, storage, embedding
        )

    async def _evict_unreplayable_cache_row(
        self,
        result: CacheResult,
        *,
        model: str | None,
        raw_scope: str | None,
        storage_scope_key: str,
    ) -> None:
        """Remove stored rows when a similarity hit cannot be serialized to clients.

        Args:
            result: Lookup outcome that was a hit but failed replay conversion.
            model: Extracted model discriminator for the bucket.
            raw_scope: Extracted scope before normalization, if any.
            storage_scope_key: Resolved scope key passed to ``SemanticCache``.
        """
        entry_id = result.cache_entry_id
        if entry_id is None:
            return
        delete_fn = getattr(self._cache, "delete_entry_by_id", None)
        if delete_fn is None:
            return
        try:
            await delete_fn(
                entry_id,
                model=model,
                scope=raw_scope,
                storage_scope_key=storage_scope_key,
            )
        except Exception:
            _logger.exception(
                "Eviction failed for unreplayable cache hit entry_id=%s",
                entry_id,
            )

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
        try:
            body = await self._read_body(receive)
        except HTTPException as exc:
            detail = exc.detail
            text = (
                detail
                if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance] -- FastAPI widens HTTPException.detail
                    detail, str
                )
                else "Request body exceeds configured maximum size."
            )
            await self._send_response(
                PlainTextResponse(text, status_code=exc.status_code),
                scope,
                send,
            )
            return
        body_replayed = False

        async def _request_receive() -> Message:
            nonlocal body_replayed
            if body_replayed:
                return {"type": "http.request", "body": b"", "more_body": False}
            body_replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive=_request_receive)

        lookup_context = await extract_lookup_context_or_passthrough(
            request=request,
            scope=scope,
            body=body,
            send=send,
            require_cache_scope=self._require_cache_scope,
            scope_settings=self._scope_settings,
            extract_query=self._extract_query,
            extract_model=self._extract_model or self._default_extract_model,
            extract_scope_required=self._extract_scope
            or self._default_extract_scope_from_request_context,
            extract_scope_optional=self._extract_scope,
            log_extraction_failure=lambda req, phase, exc: self._log_extraction_failure(
                req,
                phase=phase,
                exc=exc,
            ),
            send_passthrough_fn=lambda s, b, out_send: send_passthrough(
                scope=s,
                body=b,
                send=out_send,
                call_downstream=self._call_downstream,
                send_response=self._send_response,
            ),
        )
        if lookup_context is None:
            return
        query = lookup_context.query
        model = lookup_context.model
        raw_scope = lookup_context.raw_scope
        scope_storage = lookup_context.scope_storage

        result, cache_read_error = await self._cache_get_fail_open(
            request,
            query,
            model,
            scope=raw_scope,
            storage_scope_key=scope_storage,
            phase="preflight",
        )
        if await send_cache_hit_if_available(
            result=result,
            scope=scope,
            send=send,
            response_from_cache_hit=lambda res: self._response_from_cache_hit(
                result=res
            ),
            send_response=self._send_response,
            on_unreplayable_hit=lambda res: self._evict_unreplayable_cache_row(
                res,
                model=model,
                raw_scope=raw_scope,
                storage_scope_key=scope_storage,
            ),
        ):
            return

        flight = await self._coordination.get_flight_lock(query, model, scope_storage)
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
            if await send_cache_hit_if_available(
                result=result,
                scope=scope,
                send=send,
                response_from_cache_hit=lambda res: self._response_from_cache_hit(
                    result=res
                ),
                send_response=self._send_response,
                on_unreplayable_hit=lambda res: self._evict_unreplayable_cache_row(
                    res,
                    model=model,
                    raw_scope=raw_scope,
                    storage_scope_key=scope_storage,
                ),
            ):
                return

            if await self._coordination.upstream_blocked_by_circuit():
                await send_circuit_open_response(
                    scope=scope,
                    send=send,
                    cache_read_error=cache_read_error,
                    header_circuit=self._HEADER_CIRCUIT,
                    miss_headers=lambda read_error: self._miss_headers(
                        cache_read_error=read_error
                    ),
                    send_response=self._send_response,
                )
                return

            miss = self._miss_headers(cache_read_error=cache_read_error)
            if self._scope_settings.response_mode == "tee":
                upstream_status = await stream_tee_and_store(
                    app=self.app,
                    scope=scope,
                    body=body,
                    send=send,
                    lookup_ctx=lookup_context,
                    request=request,
                    query_embedding=result.query_embedding,
                    max_body_bytes=self._max_response_body_bytes,
                    miss_headers=miss,
                    response_allows_cache_store=self._response_allows_cache_store,
                    response_shape_allows_cache_store=self._shape_validator(),
                    cache_record_from_response=self._record_builder(),
                    cache_put=self._put_callback(),
                )
                await self._coordination.record_upstream_status_for_circuit(
                    upstream_status
                )
                return

            response = await self._call_downstream(scope, body)
            await self._coordination.record_upstream_status_for_circuit(
                response.status_code
            )
            final, payload = prepare_response_for_client(
                response=response,
                miss_headers=miss,
                merge_response_headers=self._merge_response_headers,
            )

            await maybe_store_cache_entry(
                request=request,
                body=body,
                response=response,
                payload=payload,
                query=query,
                model=model,
                raw_scope=raw_scope,
                scope_storage=scope_storage,
                query_embedding=result.query_embedding,
                response_allows_cache_store=self._response_allows_cache_store,
                response_shape_allows_cache_store=self._shape_validator(),
                cache_record_from_response=self._record_builder(),
                cache_put=self._put_callback(),
            )

            await self._send_response(final, scope, send)
            return
