"""Unit tests for ``stream_cache_hit`` ASGI chunk replay."""

from __future__ import annotations

import json

import pytest
from starlette.types import Message, Scope

from semanticcache.middleware.core.replay import stream_cache_hit
from semanticcache.types import CacheResult

_MARKER = "__semanticcache_record_v1__"


def _scope() -> Scope:
    """Build a minimal HTTP ASGI scope."""
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/test",
        "raw_path": b"/v1/test",
        "root_path": "",
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }


def _hit_result(
    body: dict[str, object],
    status: int = 200,
    extra_headers: dict[str, str] | None = None,
    similarity: float = 0.95,
) -> CacheResult:
    """Build a replayable CacheResult with the standard envelope."""
    stored_headers: dict[str, str] = {"content-type": "application/json"}
    if extra_headers:
        stored_headers.update(extra_headers)
    return CacheResult(
        is_hit=True,
        similarity=similarity,
        source="embedders.sbert",
        response={
            _MARKER: True,
            "body": body,
            "meta": {
                "status_code": status,
                "headers": stored_headers,
                "media_type": "application/json",
            },
        },
    )


async def test_stream_cache_hit_single_chunk_returns_true() -> None:
    """Single-chunk mode emits start + one body message with more_body=False."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = _hit_result({"answer": 42})
    sent = await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
        chunk_size=0,
    )

    assert sent is True
    assert len(messages) == 2

    start = messages[0]
    assert start["type"] == "http.response.start"
    assert start["status"] == 200
    header_map = {
        k.decode("latin-1").lower(): v.decode("latin-1")
        for k, v in start["headers"]
    }
    assert header_map["x-cache"] == "HIT"
    assert "content-length" not in header_map

    body_msg = messages[1]
    assert body_msg["type"] == "http.response.body"
    assert body_msg["more_body"] is False
    parsed = json.loads(body_msg["body"])
    assert parsed == {"answer": 42}


async def test_stream_cache_hit_multi_chunk_splits_body() -> None:
    """Positive chunk_size produces multiple body messages; final has more_body=False."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    body_dict = {"key": "value", "num": 1}
    result = _hit_result(body_dict)
    sent = await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
        chunk_size=5,
    )

    assert sent is True

    body_messages = [m for m in messages if m["type"] == "http.response.body"]
    assert len(body_messages) > 1

    for msg in body_messages[:-1]:
        assert msg["more_body"] is True
    assert body_messages[-1]["more_body"] is False

    reassembled = b"".join(m["body"] for m in body_messages)
    assert json.loads(reassembled) == body_dict


async def test_stream_cache_hit_no_content_length_header() -> None:
    """content-length is never emitted regardless of stored headers."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = _hit_result({"x": 1}, extra_headers={"content-length": "999"})
    await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
    )

    start = messages[0]
    header_names = {k.decode("latin-1").lower() for k, _ in start["headers"]}
    assert "content-length" not in header_names


async def test_stream_cache_hit_replays_status_code() -> None:
    """Non-200 status codes from the cache record are forwarded."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = _hit_result({"created": True}, status=201)
    await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
    )

    assert messages[0]["status"] == 201


async def test_stream_cache_hit_includes_similarity_header() -> None:
    """X-Cache-Similarity header reflects the stored similarity score."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = _hit_result({"ok": True}, similarity=0.987654)
    await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
    )

    start = messages[0]
    header_map = {
        k.decode("latin-1").lower(): v.decode("latin-1")
        for k, v in start["headers"]
    }
    assert header_map["x-cache-similarity"] == "0.987654"


async def test_stream_cache_hit_not_replayable_returns_false() -> None:
    """Missing replay envelope returns False without calling send."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = CacheResult(
        is_hit=True,
        similarity=0.9,
        source="none",
        response={"plain": "payload", "no_marker": True},
    )
    sent = await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
    )

    assert sent is False
    assert messages == []


async def test_stream_cache_hit_invalid_body_type_returns_false() -> None:
    """Replay envelope with a non-dict body returns False without calling send."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = CacheResult(
        is_hit=True,
        similarity=0.9,
        source="none",
        response={_MARKER: True, "body": ["not", "a", "dict"]},
    )
    sent = await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
    )

    assert sent is False
    assert messages == []


async def test_stream_cache_hit_strips_sensitive_headers() -> None:
    """Security-sensitive headers from the stored record are never forwarded."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    result = _hit_result(
        {"ok": True},
        extra_headers={
            "set-cookie": "session=abc",
            "authorization": "Bearer tok",
            "x-safe": "keep",
        },
    )
    await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
    )

    start = messages[0]
    header_names = {k.decode("latin-1").lower() for k, _ in start["headers"]}
    assert "set-cookie" not in header_names
    assert "authorization" not in header_names
    assert "x-safe" in header_names


async def test_stream_cache_hit_chunk_size_equal_to_body_length_is_single_chunk() -> None:
    """chunk_size exactly equal to body byte length produces one chunk."""
    messages: list[Message] = []

    async def capture(msg: Message) -> None:
        messages.append(msg)

    body_dict = {"z": 1}
    result = _hit_result(body_dict)
    body_bytes = json.dumps(body_dict, ensure_ascii=False).encode("utf-8")

    await stream_cache_hit(
        result=result,
        cache_record_marker=_MARKER,
        cache_header_name="X-Cache",
        source_header_name="X-Cache-Source",
        similarity_header_name="X-Cache-Similarity",
        send=capture,
        scope=_scope(),
        chunk_size=len(body_bytes),
    )

    body_messages = [m for m in messages if m["type"] == "http.response.body"]
    assert len(body_messages) == 1
    assert body_messages[0]["more_body"] is False
