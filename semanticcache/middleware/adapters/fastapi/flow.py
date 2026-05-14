"""Orchestration helpers for the FastAPI middleware adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Scope, Send

from ....cache import resolve_cache_scope
from ....types import CacheResult

from .asgi_io import TeeSend

if TYPE_CHECKING:
    from ....config import CacheSettings

_logger = logging.getLogger(__name__)


def normalize_request_path(path: str) -> str:
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


def compose_cache_lookup_query(
    *,
    method: str,
    normalized_path: str,
    model: str | None,
    semantic_query: str,
) -> str:
    """Build middleware cache lookup text with route and model dimensions.

    Args:
        method: Uppercase HTTP method.
        normalized_path: Normalized request path.
        model: Optional model discriminator.
        semantic_query: Extracted semantic lookup text.

    Returns:
        Stable lookup text that scopes semantic similarity by endpoint context.
    """
    model_value = (model or "").strip() or "-"
    return (
        f"method={method}\n"
        f"path={normalized_path}\n"
        f"model={model_value}\n"
        f"query={semantic_query.strip()}"
    )


@dataclass(frozen=True, slots=True)
class LookupContext:
    """Hold extracted lookup inputs for one cacheable request."""

    query: str
    model: str | None
    raw_scope: str | None
    scope_storage: str


async def send_passthrough(
    *,
    scope: Scope,
    body: bytes,
    send: Send,
    call_downstream: Callable[[Scope, bytes], Awaitable[Response]],
    send_response: Callable[[Response, Scope, Send], Awaitable[None]],
) -> None:
    """Call downstream and emit the response unchanged.

    Args:
        scope: Current request ASGI scope.
        body: Buffered request body.
        send: ASGI send callable.
        call_downstream: Downstream invoker callback.
        send_response: Response emitter callback.
    """
    passthrough = await call_downstream(scope, body)
    await send_response(passthrough, scope, send)


async def send_cache_hit_if_available(
    *,
    result: CacheResult,
    scope: Scope,
    send: Send,
    response_from_cache_hit: Callable[[CacheResult], Response | None],
    send_response: Callable[[Response, Scope, Send], Awaitable[None]],
    on_unreplayable_hit: Callable[[CacheResult], Awaitable[None]] | None = None,
    stream_cache_hit: (
        Callable[[CacheResult, Send, Scope], Awaitable[bool]] | None
    ) = None,
) -> bool:
    """Send cached response when replayable.

    Args:
        result: Cache lookup result.
        scope: Current request ASGI scope.
        send: ASGI send callable.
        response_from_cache_hit: Cache replay builder callback. Used when
            ``stream_cache_hit`` is not provided.
        send_response: Response emitter callback. Used when
            ``stream_cache_hit`` is not provided.
        on_unreplayable_hit: Optional hook when ``is_hit`` is True but the
            payload cannot be replayed (for example corrupt ``body``).
        stream_cache_hit: Optional async callable that emits the cached
            response directly as raw ASGI messages. When provided it is called
            instead of ``response_from_cache_hit`` + ``send_response``. Returns
            ``True`` when the response was emitted, ``False`` when the record is
            not replayable (triggering the same unreplayable-hit path).

    Returns:
        True when a cached response was sent.
    """
    if not result.is_hit:
        return False

    if stream_cache_hit is not None:
        streamed = await stream_cache_hit(result, send, scope)
        if not streamed:
            payload = result.response
            detail = "response_missing"
            if isinstance(payload, dict):
                body_obj = payload.get("body")
                detail: str = f"body_type={type(body_obj).__name__}"
            _logger.warning(
                "Semantic cache vector hit is not replayable; treating as miss. "
                "similarity=%s source=%s cache_entry_id=%s detail=%s",
                result.similarity,
                result.source,
                result.cache_entry_id,
                detail,
            )
            if on_unreplayable_hit is not None:
                await on_unreplayable_hit(result)
            return False
        return True

    cached_response = response_from_cache_hit(result)
    if cached_response is None:
        payload = result.response
        detail = "response_missing"
        if isinstance(payload, dict):
            body_obj: object = payload.get("body")
            detail = f"body_type={type(body_obj).__name__}"
        _logger.warning(
            "Semantic cache vector hit is not replayable; treating as miss. "
            "similarity=%s source=%s cache_entry_id=%s detail=%s",
            result.similarity,
            result.source,
            result.cache_entry_id,
            detail,
        )
        if on_unreplayable_hit is not None:
            await on_unreplayable_hit(result)
        return False
    await send_response(cached_response, scope, send)
    return True


async def send_circuit_open_response(
    *,
    scope: Scope,
    send: Send,
    cache_read_error: bool,
    header_circuit: str,
    miss_headers: Callable[[bool], dict[str, str]],
    send_response: Callable[[Response, Scope, Send], Awaitable[None]],
) -> None:
    """Send 503 response when the 429 circuit breaker is open.

    Args:
        scope: Current request ASGI scope.
        send: ASGI send callable.
        cache_read_error: Whether to include cache read error response header.
        header_circuit: Header key for circuit status.
        miss_headers: Miss header builder callback.
        send_response: Response emitter callback.
    """
    miss = miss_headers(cache_read_error)
    circuit_headers = {
        **miss,
        header_circuit: "OPEN",
    }
    await send_response(
        JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Upstream is temporarily not contacted after repeated "
                    "HTTP 429 responses; only cache hits are served until the "
                    "cooldown elapses."
                )
            },
            headers=circuit_headers,
        ),
        scope,
        send,
    )


def prepare_response_for_client(
    *,
    response: Response,
    miss_headers: Mapping[str, str],
    merge_response_headers: Callable[[Response, Mapping[str, str]], None],
) -> tuple[Response, dict[str, object] | None]:
    """Build final client response and optional cache payload.

    Args:
        response: Downstream response.
        miss_headers: Headers that mark cache miss metadata.
        merge_response_headers: Header merge callback.

    Returns:
        Tuple `(final_response, payload_for_cache_or_none)`.
    """
    if not (200 <= response.status_code < 300):
        merge_response_headers(response, miss_headers)
        return (response, None)

    raw_body = getattr(response, "body", None)
    if not isinstance(raw_body, bytes) or not raw_body:
        merge_response_headers(response, miss_headers)
        return (response, None)

    buffered = raw_body
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
        merge_response_headers(out, miss_headers)
        return (out, None)

    final = Response(
        content=buffered,
        status_code=response.status_code,
        headers={**dict(response.headers), **miss_headers},
        media_type=response.media_type,
        background=response.background,
    )
    if isinstance(payload, dict):
        payload_data = cast(dict[str, object], payload)
        return (final, payload_data)
    return (final, None)


async def extract_lookup_context_or_passthrough(
    *,
    request: Request,
    scope: Scope,
    body: bytes,
    send: Send,
    require_cache_scope: bool,
    scope_settings: CacheSettings,
    extract_query: Callable[[Request, bytes], Awaitable[str | None]],
    extract_model: Callable[[Request, bytes], Awaitable[str | None]],
    extract_scope_required: Callable[[Request, bytes], Awaitable[str | None]],
    extract_scope_optional: Callable[[Request, bytes], Awaitable[str | None]] | None,
    log_extraction_failure: Callable[[Request, str, Exception], None],
    send_passthrough_fn: Callable[[Scope, bytes, Send], Awaitable[None]],
) -> LookupContext | None:
    """Extract cache lookup inputs, or emit pass-through when unavailable.

    Args:
        request: Current Starlette request.
        scope: Current request ASGI scope.
        body: Buffered request body.
        send: ASGI send callable.
        require_cache_scope: Whether scope is mandatory for caching.
        scope_settings: Scope settings passed to `resolve_cache_scope`.
        extract_query: Query extractor callback.
        extract_model: Model extractor callback.
        extract_scope_required: Scope extractor used when scope is required.
        extract_scope_optional: Optional scope extractor when scope is not required.
        log_extraction_failure: Extraction failure logger callback.
        send_passthrough_fn: Pass-through response callback.

    Returns:
        Lookup context when cache lookup can proceed, otherwise None.
    """
    try:
        semantic_query = await extract_query(request, body)
    except Exception as exc:
        log_extraction_failure(request, "extract_query", exc)
        await send_passthrough_fn(scope, body, send)
        return None

    if semantic_query is None or not str(semantic_query).strip():
        await send_passthrough_fn(scope, body, send)
        return None

    try:
        model = await extract_model(request, body)
    except Exception as exc:
        log_extraction_failure(request, "extract_model", exc)
        await send_passthrough_fn(scope, body, send)
        return None

    normalized_path = normalize_request_path(request.url.path)
    query = compose_cache_lookup_query(
        method=request.method.upper(),
        normalized_path=normalized_path,
        model=model,
        semantic_query=semantic_query,
    )

    raw_scope: str | None = None
    if require_cache_scope:
        try:
            raw_scope = await extract_scope_required(request, body)
        except Exception as exc:
            log_extraction_failure(request, "extract_scope", exc)
            await send_passthrough_fn(scope, body, send)
            return None
    elif extract_scope_optional is not None:
        try:
            raw_scope = await extract_scope_optional(request, body)
        except Exception as exc:
            log_extraction_failure(request, "extract_scope", exc)
            await send_passthrough_fn(scope, body, send)
            return None

    scope_storage = resolve_cache_scope(raw_scope, settings=scope_settings)
    if scope_storage is None:
        await send_passthrough_fn(scope, body, send)
        return None
    return LookupContext(
        query=query,
        model=model,
        raw_scope=raw_scope,
        scope_storage=scope_storage,
    )


async def maybe_store_cache_entry(
    *,
    request: Request,
    body: bytes,
    response: Response,
    payload: dict[str, object] | None,
    query: str,
    model: str | None,
    raw_scope: str | None,
    scope_storage: str,
    query_embedding: list[float] | None,
    response_allows_cache_store: Callable[[Response], bool],
    response_shape_allows_cache_store: Callable[
        [Request, bytes, Response, dict[str, object], str | None, str | None],
        Awaitable[bool],
    ],
    cache_record_from_response: Callable[
        [dict[str, object], Response], dict[str, object]
    ],
    cache_put: Callable[
        [str, dict[str, object], str | None, str, list[float] | None], Awaitable[None]
    ],
) -> None:
    """Store cache entry when response and payload pass cacheability checks.

    Args:
        request: Current Starlette request.
        body: Buffered request body.
        response: Upstream response used for cache policy checks.
        payload: Parsed JSON dictionary payload, when available.
        query: Cache lookup query text.
        model: Optional model discriminator.
        raw_scope: Optional extracted scope value for validation context.
        scope_storage: Resolved storage scope key for cache writes.
        query_embedding: Optional embedding computed during cache lookup miss path.
        response_allows_cache_store: Response policy callback.
        response_shape_allows_cache_store: Response validator callback.
        cache_record_from_response: Cache record builder callback.
        cache_put: Cache write callback.
    """
    if payload is None:
        return
    if not response_allows_cache_store(response):
        return
    is_valid = await response_shape_allows_cache_store(
        request, body, response, payload, model, raw_scope
    )
    if not is_valid:
        return
    try:
        cache_record = cache_record_from_response(payload, response)
        await cache_put(query, cache_record, model, scope_storage, query_embedding)
    except Exception:
        _logger.exception(
            "Semantic cache write failed; returning upstream response unchanged."
        )


async def stream_tee_and_store(
    *,
    app: ASGIApp,
    scope: Scope,
    body: bytes,
    send: Send,
    lookup_ctx: LookupContext,
    request: Request,
    query_embedding: list[float] | None,
    max_body_bytes: int | None,
    upstream_timeout_seconds: float | None,
    response_allows_cache_store: Callable[[Response], bool],
    response_shape_allows_cache_store: Callable[
        [Request, bytes, Response, dict[str, object], str | None, str | None],
        Awaitable[bool],
    ],
    cache_record_from_response: Callable[
        [dict[str, object], Response], dict[str, object]
    ],
    miss_headers: Mapping[str, str],
    cache_put: Callable[
        [str, dict[str, object], str | None, str, list[float] | None], Awaitable[None]
    ],
) -> int:
    """Stream downstream response through a tee, then store in a background task.

    Forwards each ASGI body chunk to ``send`` immediately so the client sees
    streaming behavior, accumulates bytes in ``TeeSend``, then schedules
    ``maybe_store_cache_entry`` after the stream finishes. If the tee buffer
    exceeds ``max_body_bytes``, the client still receives the full stream but
    cache storage is skipped.

    When ``upstream_timeout_seconds`` is set and the downstream app does not
    complete within that budget, ``asyncio.TimeoutError`` is caught, a 504
    status is returned, and the flight lock is released promptly rather than
    being held for the full stall duration.

    Args:
        app: Downstream ASGI application.
        scope: Current request ASGI scope.
        body: Buffered request body replayed to the downstream app.
        send: ASGI send callable for the client connection.
        lookup_ctx: Extracted lookup text, model, and scope for cache writes.
        request: Current Starlette request.
        query_embedding: Optional embedding from the miss path for ``cache_put``.
        max_body_bytes: Maximum bytes to retain for cache assembly; None disables
            the cap.
        upstream_timeout_seconds: Optional cap on the upstream ASGI call duration.
            When exceeded the function returns 504 without completing cache storage.
            None disables the cap.
        miss_headers: Headers merged into ``http.response.start`` (for example
            ``X-Cache: MISS``).
        response_allows_cache_store: Header and policy gate for storing responses.
        response_shape_allows_cache_store: Async validator for parsed JSON payload.
        cache_record_from_response: Maps payload and response to a cache record.
        cache_put: Persists the cache record.

    Returns:
        HTTP status code observed on the downstream ``http.response.start``,
        or 504 when ``upstream_timeout_seconds`` is exceeded.
    """
    body_sent = False

    async def replay_receive() -> Message:
        nonlocal body_sent
        if body_sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    tee = TeeSend(
        real_send=send,
        max_body_bytes=max_body_bytes,
        merge_into_start=miss_headers,
    )
    try:
        if upstream_timeout_seconds is not None:
            await asyncio.wait_for(
                app(scope, replay_receive, tee),
                timeout=upstream_timeout_seconds,
            )
        else:
            await app(scope, replay_receive, tee)
    except asyncio.TimeoutError:
        _logger.warning(
            "Upstream tee call exceeded upstream_timeout_seconds=%.1f; "
            "releasing flight lock. path=%s",
            upstream_timeout_seconds,
            scope.get("path", "?"),
        )
        return 504

    if tee.over_limit:
        _logger.warning(
            "Tee buffer exceeded max_body_bytes; skipping cache store. path=%s",
            scope.get("path", "?"),
        )
        return tee.status_code

    assembled = Response(
        content=tee.body,
        status_code=tee.status_code,
        headers=tee.headers,
    )

    async def _store() -> None:
        try:
            payload_obj: object = json.loads(tee.body) if tee.body else None
        except json.JSONDecodeError:
            return
        if not isinstance(payload_obj, dict):
            return
        payload = cast(dict[str, object], payload_obj)
        await maybe_store_cache_entry(
            request=request,
            body=body,
            response=assembled,
            payload=payload,
            query=lookup_ctx.query,
            model=lookup_ctx.model,
            raw_scope=lookup_ctx.raw_scope,
            scope_storage=lookup_ctx.scope_storage,
            query_embedding=query_embedding,
            response_allows_cache_store=response_allows_cache_store,
            response_shape_allows_cache_store=response_shape_allows_cache_store,
            cache_record_from_response=cache_record_from_response,
            cache_put=cache_put,
        )

    def _on_store_done(task: asyncio.Task[None]) -> None:
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            _logger.exception(
                "Background tee cache store raised unexpectedly. path=%s",
                scope.get("path", "?"),
                exc_info=exc,
            )

    task = asyncio.create_task(_store())
    task.add_done_callback(_on_store_done)
    return tee.status_code
