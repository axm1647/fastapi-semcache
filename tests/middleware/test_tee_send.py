"""Unit tests for ``TeeSend`` ASGI streaming tee."""

from __future__ import annotations

import pytest

from semanticcache.middleware.adapters.fastapi.asgi_io import TeeSend


@pytest.mark.asyncio
async def test_tee_forwards_three_body_chunks_and_assembles_body() -> None:
    """All body messages reach ``real_send`` in order; ``body`` matches join."""
    received: list[bytes] = []

    async def fake_send(msg: dict) -> None:
        if msg["type"] == "http.response.body":
            chunk = msg.get("body", b"")
            if isinstance(chunk, bytes):
                received.append(chunk)

    tee = TeeSend(real_send=fake_send, max_body_bytes=None)
    await tee({"type": "http.response.start", "status": 200, "headers": []})
    await tee(
        {"type": "http.response.body", "body": b"a", "more_body": True},
    )
    await tee(
        {"type": "http.response.body", "body": b"b", "more_body": True},
    )
    await tee(
        {"type": "http.response.body", "body": b"c", "more_body": False},
    )

    assert received == [b"a", b"b", b"c"]
    assert tee.body == b"abc"
    assert not tee.over_limit


@pytest.mark.asyncio
async def test_tee_over_limit_clears_buffer_but_forwards_all_chunks() -> None:
    """When buffer cap is exceeded, client still receives every chunk."""
    received: list[bytes] = []

    async def fake_send(msg: dict) -> None:
        if msg["type"] == "http.response.body":
            chunk = msg.get("body", b"")
            if isinstance(chunk, bytes):
                received.append(chunk)

    tee = TeeSend(real_send=fake_send, max_body_bytes=5)
    await tee({"type": "http.response.start", "status": 200, "headers": []})
    await tee({"type": "http.response.body", "body": b"123456", "more_body": False})

    assert received == [b"123456"]
    assert tee.over_limit is True
    assert tee.body == b""


@pytest.mark.asyncio
async def test_tee_response_start_forwards_and_sets_status_and_headers() -> None:
    """``http.response.start`` is forwarded and metadata is captured."""
    forwarded: list[dict] = []

    async def fake_send(msg: dict) -> None:
        forwarded.append(msg)

    tee = TeeSend(real_send=fake_send, max_body_bytes=None)
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"x-custom", b"yes"),
    ]
    await tee(
        {
            "type": "http.response.start",
            "status": 201,
            "headers": raw_headers,
        },
    )

    assert len(forwarded) == 1
    assert forwarded[0]["type"] == "http.response.start"
    assert forwarded[0]["status"] == 201
    assert tee.status_code == 201
    assert tee.headers["content-type"] == "application/json"
    assert tee.headers["x-custom"] == "yes"


@pytest.mark.asyncio
async def test_tee_merge_into_start_headers_on_wire_and_in_tee_headers() -> None:
    """``merge_into_start`` is appended to the forwarded start message."""
    forwarded: list[dict] = []

    async def fake_send(msg: dict) -> None:
        forwarded.append(msg)

    tee = TeeSend(
        real_send=fake_send,
        max_body_bytes=None,
        merge_into_start={"X-Cache": "MISS"},
    )
    await tee(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        },
    )

    assert len(forwarded) == 1
    hdrs = forwarded[0].get("headers", [])
    assert (b"content-type", b"application/json") in hdrs
    assert (b"X-Cache", b"MISS") in hdrs
    assert tee.headers.get("X-Cache") == "MISS"
    assert tee.headers.get("content-type") == "application/json"
