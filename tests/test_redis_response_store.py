"""Tests for ``RedisResponseStore`` Redis client construction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from semanticcache.stores.response.redis_store import RedisResponseStore


@pytest.mark.asyncio
async def test_open_passes_socket_timeouts_to_from_url() -> None:
    """Pass socket timeouts through to ``redis.asyncio.from_url`` when set."""
    mock_mod = MagicMock()
    mock_client = MagicMock()
    mock_mod.from_url.return_value = mock_client

    with patch(
        "semanticcache.stores.response.redis_store._require_redis",
        return_value=mock_mod,
    ):
        store = RedisResponseStore(
            "redis://localhost:6379/0",
            socket_timeout_seconds=5.0,
            socket_connect_timeout_seconds=4.0,
        )
        await store.open()

    mock_mod.from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=4.0,
    )


@pytest.mark.asyncio
async def test_open_omits_socket_kwargs_when_unset() -> None:
    """Leave socket timeouts unset when no values are provided."""
    mock_mod = MagicMock()
    mock_client = MagicMock()
    mock_mod.from_url.return_value = mock_client

    with patch(
        "semanticcache.stores.response.redis_store._require_redis",
        return_value=mock_mod,
    ):
        store = RedisResponseStore("redis://localhost:6379/0")
        await store.open()

    mock_mod.from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        decode_responses=True,
    )
