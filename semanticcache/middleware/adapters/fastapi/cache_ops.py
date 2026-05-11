"""Cache and store-policy helpers for FastAPI semantic cache middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import TYPE_CHECKING

from starlette.responses import Response

from ....types import CacheResult

if TYPE_CHECKING:
    from .middleware import ResponseValidationContext


async def cache_get_fail_open(
    *,
    cache_get: Callable[[str, str | None, str], Awaitable[CacheResult]],
    query: str,
    model: str | None,
    scope: str | None,
    storage_scope_key: str,
    on_failure: Callable[[str, str | None, str | None, str, Exception], None],
    phase: str,
) -> tuple[CacheResult, bool]:
    """Run cache.get and map failures to a miss without raising.

    Args:
        cache_get: Cache read callback.
        query: Cache lookup text.
        model: Optional embedder routing key.
        scope: Optional tenant or namespace for logs.
        storage_scope_key: Resolved storage scope key for cache read.
        on_failure: Failure callback for logging.
        phase: Log label for this read (preflight vs double_check).

    Returns:
        Tuple `(result, cache_read_error)` where read errors become synthetic misses.
    """
    try:
        return (await cache_get(query, model, storage_scope_key), False)
    except Exception as exc:
        on_failure(query, model, scope, phase, exc)
        return (CacheResult(is_hit=False), True)


def response_allows_cache_store(response: Response) -> bool:
    """Return True when upstream response headers permit cache storage.

    Args:
        response: Upstream response candidate for cache persistence.

    Returns:
        False when Cache-Control has no-store or private, or Set-Cookie is present.
    """
    # Use raw ASGI headers consistently for both Set-Cookie and Cache-Control checks
    # to avoid drift if Starlette's header mapping behavior diverges from raw_headers.
    cache_control_value: bytes | None = None
    for name, value in response.raw_headers:
        lower_name = name.lower()
        if lower_name == b"set-cookie":
            return False
        if lower_name == b"cache-control" and cache_control_value is None:
            cache_control_value = value
    if cache_control_value is None:
        return True
    cache_control = cache_control_value.decode("latin-1", "ignore")
    directives = {
        part.strip().lower() for part in cache_control.split(",") if part.strip()
    }
    return "no-store" not in directives and "private" not in directives


async def response_shape_allows_cache_store(
    *,
    context: ResponseValidationContext,
    validate_response: (
        Callable[[ResponseValidationContext], bool | Awaitable[bool]] | None
    ),
    on_validation_failure: Callable[[ResponseValidationContext, Exception], None],
    on_validation_rejected: Callable[[ResponseValidationContext], None],
) -> bool:
    """Return True when optional response validator accepts the payload.

    Args:
        context: Response details and parsed JSON payload to validate.
        validate_response: Optional response validation callback.
        on_validation_failure: Callback when validator raises.
        on_validation_rejected: Callback when validator returns False.

    Returns:
        True when no validator is configured or validation passes.
    """
    if validate_response is None:
        return True
    try:
        result = validate_response(context)
        if isawaitable(result):
            result = await result
    except Exception as exc:
        on_validation_failure(context, exc)
        return False
    if result:
        return True
    on_validation_rejected(context)
    return False
