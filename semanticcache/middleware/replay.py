"""Replay and response-shaping helpers for semantic cache middleware."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from starlette.responses import JSONResponse, Response

from ..types import CacheResult


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
        Response with original status and headers when metadata exists.
        Returns None when hit payload is not replayable.
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
                    replay_headers[key] = value
        headers = {**replay_headers, **hit_headers}
        return JSONResponse(
            content=body,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )
    return JSONResponse(
        content=cached_payload,
        headers=hit_headers,
    )
