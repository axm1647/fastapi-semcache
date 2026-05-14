"""Replay and response-shaping helpers for semantic cache middleware."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from starlette.responses import JSONResponse, Response
from starlette.types import Scope, Send

from ...types import CacheResult

_SKIP_HEADERS: frozenset[str] = frozenset(
    {"set-cookie", "authorization", "www-authenticate", "proxy-authenticate"}
)


def build_hit_headers(
    *,
    result: CacheResult,
    cache_header_name: str,
    source_header_name: str,
    similarity_header_name: str,
) -> dict[str, str]:
    """Build response headers for a cache hit.

    Args:
        result: Successful lookup result.
        cache_header_name: Header key that marks hit or miss outcomes.
        source_header_name: Header key for the cache source name.
        similarity_header_name: Header key for similarity metadata.

    Returns:
        Header map including cache hit metadata.
    """
    headers: dict[str, str] = {
        cache_header_name: "HIT",
        source_header_name: result.source,
    }
    if result.similarity is not None:
        headers[similarity_header_name] = f"{result.similarity:.6f}"
    return headers


def build_miss_headers(
    *,
    cache_header_name: str,
    cache_error_header_name: str,
    cache_read_error: bool = False,
) -> dict[str, str]:
    """Build response headers for uncached or pass-through responses.

    Args:
        cache_header_name: Header key that marks hit or miss outcomes.
        cache_error_header_name: Header key for cache read failure marker.
        cache_read_error: When True, include the cache read error marker.

    Returns:
        Header map with cache miss metadata.
    """
    headers: dict[str, str] = {cache_header_name: "MISS"}
    if cache_read_error:
        headers[cache_error_header_name] = "1"
    return headers


def merge_response_headers(response: Response, extra: Mapping[str, str]) -> None:
    """Merge additional headers into a response in place.

    Args:
        response: ASGI response whose headers are mutated.
        extra: Additional header keys and values.
    """
    for key, value in extra.items():
        response.headers[key] = value


def cache_record_from_response(
    *,
    payload: dict[str, object],
    response: Response,
    cache_record_marker: str,
) -> dict[str, object]:
    """Build a cache record with payload and response replay metadata.

    Security-sensitive headers (``set-cookie``, ``authorization``,
    ``www-authenticate``, ``proxy-authenticate``) are stripped before storage
    so they are never persisted to Postgres or Redis and cannot be replayed to
    unrelated clients.

    Args:
        payload: Parsed JSON object body from the upstream response.
        response: Upstream response to mirror on future cache hits.
        cache_record_marker: Marker key used to identify replayable records.

    Returns:
        Cache record dictionary with JSON body plus replay metadata.
    """
    headers_to_store: dict[str, str] = {}
    for key, value in response.headers.items():
        lower = key.lower()
        if lower == "content-length":
            continue
        if lower.startswith("x-cache"):
            continue
        if lower in _SKIP_HEADERS:
            continue
        headers_to_store[key] = value
    return {
        cache_record_marker: True,
        "body": payload,
        "meta": {
            "status_code": response.status_code,
            "headers": headers_to_store,
            "media_type": response.media_type,
        },
    }


def response_from_cache_hit(
    *,
    result: CacheResult,
    cache_record_marker: str,
    cache_header_name: str,
    source_header_name: str,
    similarity_header_name: str,
) -> Response | None:
    """Convert a cache hit result to the HTTP response sent to clients.

    Args:
        result: Cache lookup output with payload and similarity metadata.
        cache_record_marker: Marker key used to identify replayable records.
        cache_header_name: Header key that marks hit or miss outcomes.
        source_header_name: Header key for the cache source name.
        similarity_header_name: Header key for similarity metadata.

    Returns:
        Response with original status and headers when the stored record uses the
        replay envelope (marker key plus ``body`` and ``meta``).

        Returns None when the hit payload is not replayable (missing envelope or
        invalid ``body``), so status and headers cannot be reconstructed safely.
    """
    cached_payload = result.response
    if cached_payload is None:
        return None
    hit_headers = build_hit_headers(
        result=result,
        cache_header_name=cache_header_name,
        source_header_name=source_header_name,
        similarity_header_name=similarity_header_name,
    )
    if cached_payload.get(cache_record_marker) is True:
        body_obj: object = cached_payload.get("body")
        if not isinstance(body_obj, dict):
            return None
        body = cast(dict[str, object], body_obj)
        raw_meta: object = cached_payload.get("meta")
        meta = cast(dict[str, object], raw_meta) if isinstance(raw_meta, dict) else {}
        status_code_raw: object = meta.get("status_code", 200)
        status_code = int(status_code_raw) if isinstance(status_code_raw, int) else 200
        media_type_raw: object = meta.get("media_type")
        media_type = media_type_raw if isinstance(media_type_raw, str) else None
        headers_raw: object = meta.get("headers")
        replay_headers: dict[str, str] = {}
        if isinstance(headers_raw, dict):
            headers_data = cast(dict[object, object], headers_raw)
            for key, value in headers_data.items():
                if isinstance(key, str) and isinstance(value, str):
                    if key.lower() not in _SKIP_HEADERS:
                        replay_headers[key] = value
        headers = {**replay_headers, **hit_headers}
        return JSONResponse(
            content=body,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )
    return None


async def stream_cache_hit(
    *,
    result: CacheResult,
    cache_record_marker: str,
    cache_header_name: str,
    source_header_name: str,
    similarity_header_name: str,
    send: Send,
    scope: Scope,
    chunk_size: int = 0,
) -> bool:
    """Emit a cached hit response as raw ASGI body chunks.

    Sends ``http.response.start`` followed by one or more ``http.response.body``
    messages directly to ``send``, without constructing a ``Response`` object.
    This produces the same ASGI framing as a streaming miss response: no
    ``content-length`` header is emitted, and ``more_body`` is ``True`` for all
    but the final chunk.

    Args:
        result: Cache lookup output with payload and similarity metadata.
        cache_record_marker: Marker key used to identify replayable records.
        cache_header_name: Header key that marks hit or miss outcomes.
        source_header_name: Header key for the cache source name.
        similarity_header_name: Header key for similarity metadata.
        send: ASGI send callable for the current client connection.
        scope: Current request ASGI scope (unused at runtime but kept for
            symmetry with other ASGI helpers and future extension).
        chunk_size: Maximum byte length of each emitted body chunk. ``0``
            (default) sends the full body as a single chunk. Positive values
            split the serialised body into sequential chunks of at most this
            many bytes, which allows clients that measure time-to-first-byte
            or process tokens incrementally to observe progressive delivery.

    Returns:
        True when the cached response was emitted to ``send``.
        False when the hit payload is not replayable (missing replay envelope
        or invalid ``body``), so the caller should fall back to a miss.
    """
    cached_payload = result.response
    if cached_payload is None:
        return False

    if cached_payload.get(cache_record_marker) is not True:
        return False

    body_obj: object = cached_payload.get("body")
    if not isinstance(body_obj, dict):
        return False
    body = cast(dict[str, object], body_obj)

    raw_meta: object = cached_payload.get("meta")
    meta = cast(dict[str, object], raw_meta) if isinstance(raw_meta, dict) else {}

    status_code_raw: object = meta.get("status_code", 200)
    status_code = int(status_code_raw) if isinstance(status_code_raw, int) else 200

    headers_raw: object = meta.get("headers")
    replay_headers: dict[str, str] = {}
    if isinstance(headers_raw, dict):
        headers_data = cast(dict[object, object], headers_raw)
        for key, value in headers_data.items():
            if isinstance(key, str) and isinstance(value, str):
                lower = key.lower()
                if lower not in _SKIP_HEADERS and lower != "content-length":
                    replay_headers[key] = value

    hit_headers = build_hit_headers(
        result=result,
        cache_header_name=cache_header_name,
        source_header_name=source_header_name,
        similarity_header_name=similarity_header_name,
    )
    merged_headers = {**replay_headers, **hit_headers}

    raw_headers: list[tuple[bytes, bytes]] = [
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in merged_headers.items()
    ]
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": raw_headers,
        }
    )

    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    if chunk_size <= 0 or len(body_bytes) <= chunk_size:
        await send(
            {"type": "http.response.body", "body": body_bytes, "more_body": False}
        )
        return True

    offset = 0
    total = len(body_bytes)
    while offset < total:
        chunk = body_bytes[offset : offset + chunk_size]
        offset += chunk_size
        more_body = offset < total
        await send(
            {"type": "http.response.body", "body": chunk, "more_body": more_body}
        )

    return True
