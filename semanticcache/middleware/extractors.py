"""Request and JSON extraction helpers for semantic cache middleware."""

from __future__ import annotations

import json
from typing import cast

from starlette.requests import Request


def _extract_query_from_mapping(data: dict[str, object]) -> str | None:
    """Pick a cache key string from common LLM and search JSON shapes.

    Args:
        data: Parsed JSON object body.

    Returns:
        Non-empty query text, or None if no usable field was found.
    """
    for key in ("query", "prompt", "input"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    messages_obj: object = data.get("messages")
    if isinstance(messages_obj, list):
        messages = cast(list[object], messages_obj)
        parts: list[str] = []
        for item_obj in messages:
            if not isinstance(item_obj, dict):
                continue
            item = cast(dict[str, object], item_obj)
            if item.get("role") != "user":
                continue
            content_obj: object = item.get("content")
            if isinstance(content_obj, str) and content_obj.strip():
                parts.append(content_obj)
            elif isinstance(content_obj, list):
                content_blocks = cast(list[object], content_obj)
                for block_obj in content_blocks:
                    if not isinstance(block_obj, dict):
                        continue
                    block = cast(dict[str, object], block_obj)
                    if block.get("type") == "text":
                        text_obj: object = block.get("text")
                        if isinstance(text_obj, str) and text_obj.strip():
                            parts.append(text_obj)
        if parts:
            return "\n".join(parts)
    return None


def _json_scope_field_value(value: object) -> str | None:
    """Normalize cache scope JSON values for cache isolation.

    Args:
        value: Raw JSON field value.

    Returns:
        Non-empty scope string, or None when the value is unusable.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    return None


async def default_extract_query(request: Request, body: bytes) -> str | None:
    """Derive cache lookup text from JSON query, prompt, messages, and input.

    Args:
        request: Incoming ASGI request used for Content-Type detection.
        body: Raw body bytes already read from the stream.

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
        parsed_obj: object = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_obj, dict):
        parsed = cast(dict[str, object], parsed_obj)
        return _extract_query_from_mapping(parsed)
    return None


async def default_extract_model(
    request: Request,
    body: bytes,
    *,
    model_header_name: str,
) -> str | None:
    """Read model from header or JSON body.

    Args:
        request: Current request.
        body: Raw body bytes.
        model_header_name: Header name checked before JSON fallback.

    Returns:
        Model name if present, else None.
    """
    header_value = request.headers.get(model_header_name)
    if isinstance(header_value, str) and header_value.strip():
        return header_value.strip()
    if not body.strip():
        return None
    try:
        parsed_obj: object = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_obj, dict):
        parsed = cast(dict[str, object], parsed_obj)
        model = parsed.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None


async def default_extract_scope(
    request: Request,
    body: bytes,
    *,
    scope_header_name: str,
) -> str | None:
    """Read tenant or namespace scope from header or JSON body.

    Args:
        request: Current request.
        body: Raw body bytes.
        scope_header_name: Header name checked before JSON fallback.

    Returns:
        Non-empty scope string when present, else None.
    """
    header_value = request.headers.get(scope_header_name)
    if isinstance(header_value, str) and header_value.strip():
        return header_value.strip()
    if not body.strip():
        return None
    try:
        parsed_obj: object = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_obj, dict):
        parsed = cast(dict[str, object], parsed_obj)
        for field in ("cache_scope", "tenant_id"):
            coerced = _json_scope_field_value(parsed.get(field))
            if coerced is not None:
                return coerced
    return None
