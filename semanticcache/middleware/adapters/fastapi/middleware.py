"""FastAPI / Starlette HTTP middleware for semantic response caching."""

# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
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
            body = await read_body(receive, max_body_bytes=self._max_request_body_bytes)
        except HTTPException as exc:
            detail = exc.detail
            text = (
                detail
                if isinstance(  # pyright: ignore[reportUnnecessaryIsInstance] -- FastAPI widens HTTPException.detail
                    detail, str
                )
                else "Request body exceeds configured maximum size."
            )
            await send_response(
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

        # -- bound helpers used throughout this request --
        model_header = self._model_header_name
        scope_header = self._scope_header_name
        marker = self._CACHE_RECORD_MARKER
        validate_response = self._validate_response
        max_resp_bytes = self._max_response_body_bytes

        async def _do_cache_get(
            query: str, model: str | None, storage: str, *, phase: str, raw_scope: str | None
        ) -> tuple[CacheResult, bool]:
            return await cache_get_fail_open(
                cache_get=lambda q, m, s: self._cache.get(q, model=m, storage_scope_key=s),
                query=query,
                model=model,
                scope=raw_scope,
                storage_scope_key=storage,
                on_failure=lambda q, m, scp, ph, exc: log_cache_get_failure(
                    request=request,
                    query=q,
                    model=m,
                    scope=scp,
                    phase=ph,
                    exc=exc,
                ),
                phase=phase,
            )

        async def _shape_validator(
            req: Request,
            req_body: bytes,
            resp: Response,
            pld: dict[str, object],
            mdl: str | None,
            scp: str | None,
        ) -> bool:
            return await response_shape_allows_cache_store(
                context=ResponseValidationContext(
                    request=req,
                    request_body=req_body,
                    response=resp,
                    payload=pld,
                    model=mdl,
                    scope=scp,
                ),
                validate_response=validate_response,
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

        def _record_builder(pld: dict[str, object], resp: Response) -> dict[str, object]:
            return cache_record_from_response(
                payload=pld,
                response=resp,
                cache_record_marker=marker,
            )

        async def _put_callback(
            q: str,
            record: dict[str, object],
            mdl: str | None,
            storage: str,
            embedding: list[float] | None,
        ) -> None:
            await self._cache_put_with_optional_embedding(q, record, mdl, storage, embedding)

        # -- end bound helpers --

        lookup_context = await extract_lookup_context_or_passthrough(
            request=request,
            scope=scope,
            body=body,
            send=send,
            require_cache_scope=self._require_cache_scope,
            scope_settings=self._scope_settings,
            extract_query=self._extract_query,
            extract_model=self._extract_model or (
                lambda req, b: default_extract_model(req, b, model_header_name=model_header)
            ),
            extract_scope_required=self._extract_scope or (
                lambda req, b: default_extract_scope_from_request_context(
                    req, b, scope_header_name=scope_header
                )
            ),
            extract_scope_optional=self._extract_scope,
            log_extraction_failure=lambda req, phase, exc: log_extraction_failure(
                request=req,
                phase=phase,
                exc=exc,
            ),
            send_passthrough_fn=lambda s, b, out_send: send_passthrough(
                scope=s,
                body=b,
                send=out_send,
                call_downstream=lambda sc, bd: call_downstream(
                    self.app,
                    sc,
                    bd,
                    max_body_bytes=max_resp_bytes,
                    timeout_seconds=self._cache_settings.upstream_timeout_seconds,
                ),
                send_response=send_response,
            ),
        )
        if lookup_context is None:
            return
        query = lookup_context.query
        model = lookup_context.model
        raw_scope = lookup_context.raw_scope
        scope_storage = lookup_context.scope_storage

        result, cache_read_error = await _do_cache_get(
            query, model, scope_storage, phase="preflight", raw_scope=raw_scope
        )
        if await send_cache_hit_if_available(
            result=result,
            scope=scope,
            send=send,
            response_from_cache_hit=lambda res: response_from_cache_hit(
                result=res,
                cache_record_marker=marker,
                cache_header_name=self._HEADER_CACHE,
                source_header_name=self._HEADER_SOURCE,
                similarity_header_name=self._HEADER_SIMILARITY,
            ),
            send_response=send_response,
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
            result, inner_err = await _do_cache_get(
                query, model, scope_storage, phase="double_check", raw_scope=raw_scope
            )
            cache_read_error = cache_read_error or inner_err
            if await send_cache_hit_if_available(
                result=result,
                scope=scope,
                send=send,
                response_from_cache_hit=lambda res: response_from_cache_hit(
                    result=res,
                    cache_record_marker=marker,
                    cache_header_name=self._HEADER_CACHE,
                    source_header_name=self._HEADER_SOURCE,
                    similarity_header_name=self._HEADER_SIMILARITY,
                ),
                send_response=send_response,
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
                    miss_headers=lambda read_error: build_miss_headers(
                        cache_header_name=self._HEADER_CACHE,
                        cache_error_header_name=self._HEADER_CACHE_ERROR,
                        cache_read_error=read_error,
                    ),
                    send_response=send_response,
                )
                return

            miss = build_miss_headers(
                cache_header_name=self._HEADER_CACHE,
                cache_error_header_name=self._HEADER_CACHE_ERROR,
                cache_read_error=cache_read_error,
            )
            if self._scope_settings.response_mode == "tee":
                upstream_status = await stream_tee_and_store(
                    app=self.app,
                    scope=scope,
                    body=body,
                    send=send,
                    lookup_ctx=lookup_context,
                    request=request,
                    query_embedding=result.query_embedding,
                    max_body_bytes=max_resp_bytes,
                    upstream_timeout_seconds=self._cache_settings.upstream_timeout_seconds,
                    miss_headers=miss,
                    response_allows_cache_store=response_allows_cache_store,
                    response_shape_allows_cache_store=_shape_validator,
                    cache_record_from_response=_record_builder,
                    cache_put=_put_callback,
                )
                await self._coordination.record_upstream_status_for_circuit(
                    upstream_status
                )
                return

            response = await call_downstream(
                self.app,
                scope,
                body,
                max_body_bytes=max_resp_bytes,
                timeout_seconds=self._cache_settings.upstream_timeout_seconds,
            )
            await self._coordination.record_upstream_status_for_circuit(
                response.status_code
            )
            final, payload = prepare_response_for_client(
                response=response,
                miss_headers=miss,
                merge_response_headers=merge_response_headers,
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
                response_allows_cache_store=response_allows_cache_store,
                response_shape_allows_cache_store=_shape_validator,
                cache_record_from_response=_record_builder,
                cache_put=_put_callback,
            )

            await send_response(final, scope, send)
            return
