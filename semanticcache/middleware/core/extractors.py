"""Request and JSON extraction helpers for semantic cache middleware.

Warning:
    ``default_extract_scope_from_request_context`` trusts scope values supplied by
    the client (HTTP headers and JSON body). Use it only for single-tenant
    workloads or when a trusted gateway strips or overwrites untrusted fields
    before requests reach your app. For multi-tenant production, provide
    ``SemanticCacheMiddleware(..., extract_scope=...)`` that resolves scope from
    server-side identity (for example ``trusted_extract_scope_from_server_side``
    after authentication middleware sets ``request.state``).
"""

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
    ct = (request.headers.get("content-type") or "").lower()
    body_stripped = body.lstrip()
    if not body_stripped:
        return None

    # Prefer explicit JSON content types and only fall back to byte sniffing
    # (after stripping leading whitespace) when the Content-Type is missing or
    # clearly text-based. This avoids aggressively treating arbitrary binary
    # payloads that happen to start with "{" as JSON.
    is_json_content_type = "application/json" in ct or "+json" in ct
    looks_json = is_json_content_type or (
        (not ct or ct.startswith("text/"))
        and body_stripped[:1] in (b"{", b"[")
    )
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


async def default_extract_scope_from_request_context(
    request: Request,
    body: bytes,
    *,
    scope_header_name: str,
) -> str | None:
    """Read scope from client-visible header and JSON body fields.

    Suitable for single-tenant apps, development, or deployments where a trusted
    proxy sets scope headers or JSON from authenticated identity. Callers that
    expose this middleware directly to untrusted clients should prefer
    ``extract_scope`` implementations that derive scope only from server-side
    state (see ``trusted_extract_scope_from_server_side``).

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


async def trusted_extract_scope_from_server_side(request: Request) -> str | None:
    """Read cache partition scope only from ``request.state``.

    Set ``request.state.cache_scope`` or ``request.state.tenant_id`` in trusted
    middleware (authentication, tenancy resolution, or an edge gateway adapter)
    before ``SemanticCacheMiddleware`` runs. Integer ``tenant_id`` values are
    normalized to strings consistently with JSON extraction.

    Args:
        request: Request whose ``state`` attributes were populated by trusted code.

    Returns:
        Non-empty scope string when present, else None.
    """
    state = request.state
    for attr in ("cache_scope", "tenant_id"):
        raw: object = getattr(state, attr, None)
        coerced = _json_scope_field_value(raw)
        if coerced is not None:
            return coerced
    return None
