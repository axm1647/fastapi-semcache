"""Tests for bounded in-flight lock retention in ``SemanticCacheMiddleware``."""

# pyright: reportCallIssue=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.middleware.fastapi import SemanticCacheMiddleware


def _make_middleware(*, max_entries: int) -> SemanticCacheMiddleware:
    """Build middleware with a bounded in-flight lock table.

    Args:
        max_entries: Maximum retained lock entries.

    Returns:
        Configured middleware instance for direct unit testing.
    """
    app = FastAPI()
    return SemanticCacheMiddleware(
        app=app,
        cache=cast(SemanticCache, object()),
        cache_settings=CacheSettings(
            middleware_flight_lock_max_entries=max_entries,
        ),
    )


@pytest.mark.asyncio
async def test_get_flight_lock_evicts_oldest_unlocked_entry_when_over_capacity() -> (
    None
):
    """Evict the LRU unlocked key when inserting beyond capacity."""
    middleware = _make_middleware(max_entries=2)

    _ = await middleware._get_flight_lock("q1", "m", "")
    _ = await middleware._get_flight_lock("q2", "m", "")
    _ = await middleware._get_flight_lock("q3", "m", "")

    assert len(middleware._flight_locks) == 2
    assert ("q1", "m", "") not in middleware._flight_locks
    assert ("q2", "m", "") in middleware._flight_locks
    assert ("q3", "m", "") in middleware._flight_locks


@pytest.mark.asyncio
async def test_get_flight_lock_access_refreshes_lru_position() -> None:
    """Keep a recently reused lock and evict the older unlocked key."""
    middleware = _make_middleware(max_entries=2)

    _ = await middleware._get_flight_lock("q1", "m", "")
    _ = await middleware._get_flight_lock("q2", "m", "")
    _ = await middleware._get_flight_lock("q1", "m", "")
    _ = await middleware._get_flight_lock("q3", "m", "")

    assert len(middleware._flight_locks) == 2
    assert ("q1", "m", "") in middleware._flight_locks
    assert ("q2", "m", "") not in middleware._flight_locks
    assert ("q3", "m", "") in middleware._flight_locks


@pytest.mark.asyncio
async def test_get_flight_lock_skips_eviction_for_locked_entries() -> None:
    """Do not evict entries that are currently coordinating active requests."""
    middleware = _make_middleware(max_entries=2)

    held = await middleware._get_flight_lock("q1", "m", "")
    await held.acquire()
    try:
        _ = await middleware._get_flight_lock("q2", "m", "")
        _ = await middleware._get_flight_lock("q3", "m", "")
    finally:
        held.release()

    assert len(middleware._flight_locks) == 2
    assert ("q1", "m", "") in middleware._flight_locks
    assert ("q2", "m", "") not in middleware._flight_locks
    assert ("q3", "m", "") in middleware._flight_locks
