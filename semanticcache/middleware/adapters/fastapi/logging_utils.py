"""Logging helpers for FastAPI semantic cache middleware."""

from __future__ import annotations

import logging

from starlette.requests import Request

_CACHE_KEY_LOG_MAX = 48
_logger = logging.getLogger(__name__)


def cache_key_snippet(query: str, max_chars: int = _CACHE_KEY_LOG_MAX) -> str:
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


def request_id_for_log(request: Request) -> str | None:
    """Extract best-effort request or trace id from common headers.

    Args:
        request: Current ASGI request.

    Returns:
        First non-empty id header value capped for log safety, or None.
    """
    for name in ("X-Request-ID", "X-Correlation-ID", "X-Trace-ID"):
        raw = request.headers.get(name)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()[:128]
    return None


def log_cache_get_failure(
    *,
    request: Request,
    query: str,
    model: str | None,
    scope: str | None,
    phase: str,
    exc: Exception,
) -> None:
    """Emit a structured warning when cache read fails.

    Args:
        request: Current request for route and request id context.
        query: Cache key text.
        model: Optional model key for logs.
        scope: Optional tenant scope for logs.
        phase: Preflight or double-check phase label.
        exc: Exception raised by the cache layer.
    """
    rid = request_id_for_log(request)
    snippet = cache_key_snippet(query)
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


def log_extraction_failure(
    *,
    request: Request,
    phase: str,
    exc: Exception,
) -> None:
    """Emit a warning when query, model, or scope extraction fails.

    Args:
        request: Current request for route and request id context.
        phase: Extraction phase label.
        exc: Exception raised by the extractor.
    """
    rid = request_id_for_log(request)
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


def log_response_validation_failure(
    *,
    request: Request,
    model: str | None,
    scope: str | None,
    exc: Exception,
) -> None:
    """Emit a warning when response-shape validation raises.

    Args:
        request: Current request for route and request id context.
        model: Optional model key for logs.
        scope: Optional scope key for logs.
        exc: Exception raised by response validation callback.
    """
    rid = request_id_for_log(request)
    _logger.warning(
        "Semantic cache response validation failed: route=%s request_id=%s "
        "model=%s scope=%s error=%s: %s",
        request.url.path,
        rid if rid is not None else "-",
        (model or "").strip()[:64] or "-",
        (scope or "").strip()[:64] or "-",
        type(exc).__name__,
        exc,
        exc_info=True,
    )


def log_response_validation_rejected(
    *,
    request: Request,
    model: str | None,
    scope: str | None,
) -> None:
    """Emit a debug log when response validation rejects cache storage.

    Args:
        request: Current request for route context.
        model: Optional model key for logs.
        scope: Optional scope key for logs.
    """
    _logger.debug(
        "Semantic cache response validation rejected store: route=%s model=%s "
        "scope=%s",
        request.url.path,
        (model or "").strip()[:64] or "-",
        (scope or "").strip()[:64] or "-",
    )
