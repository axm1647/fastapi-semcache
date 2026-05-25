"""Tests for prompt-safe logging helpers."""

from __future__ import annotations

import logging

from starlette.requests import Request

from semanticcache.middleware.adapters.fastapi.logging_utils import (
    cache_key_digest,
    log_cache_get_failure,
)


def _build_request() -> Request:
    """Build a minimal request object for logging tests.

    Returns:
        Request: Starlette request with a stable path and request id header.
    """
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"x-request-id", b"req-123")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
    )


def test_cache_key_digest_is_stable_for_same_inputs() -> None:
    """Return the same digest for identical query and key inputs."""
    digest = cache_key_digest(
        "method=POST\npath=/v1/chat\nmodel=-\nquery=secret prompt",
        digest_key="unit-test-key",
    )

    assert digest == cache_key_digest(
        "method=POST\npath=/v1/chat\nmodel=-\nquery=secret prompt",
        digest_key="unit-test-key",
    )


def test_log_cache_get_failure_logs_digest_without_prompt_text(caplog) -> None:
    """Log a keyed digest instead of prompt-derived query text."""
    query = "method=POST\npath=/v1/chat\nmodel=-\nquery=secret prompt text"
    expected_digest = cache_key_digest(query, digest_key="unit-test-key")

    with caplog.at_level(
        logging.WARNING,
        logger="semanticcache.middleware.adapters.fastapi.logging_utils",
    ):
        log_cache_get_failure(
            request=_build_request(),
            query=query,
            digest_key="unit-test-key",
            model="gpt-test",
            scope="tenant-a",
            phase="preflight",
            exc=RuntimeError("boom"),
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "cache_key_digest=" in message
    assert expected_digest in message
    assert "secret prompt text" not in message
    assert "route=/v1/chat/completions" in message
    assert "request_id=req-123" in message
    assert "scope=tenant-a" in message
