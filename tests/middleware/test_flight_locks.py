"""Tests for bounded in-flight lock retention in ``SemanticCacheMiddleware``."""

# pyright: reportCallIssue=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from semanticcache.cache import SemanticCache
from semanticcache.config import CacheSettings
from semanticcache.middleware.adapters.fastapi import SemanticCacheMiddleware


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

    _ = await middleware._coordination.get_flight_lock("q1", "m", "")
    _ = await middleware._coordination.get_flight_lock("q2", "m", "")
    _ = await middleware._coordination.get_flight_lock("q3", "m", "")

    assert len(middleware._coordination._flight_locks) == 2
    assert ("q1", "m", "") not in middleware._coordination._flight_locks
    assert ("q2", "m", "") in middleware._coordination._flight_locks
    assert ("q3", "m", "") in middleware._coordination._flight_locks


@pytest.mark.asyncio
async def test_get_flight_lock_access_refreshes_lru_position() -> None:
    """Keep a recently reused lock and evict the older unlocked key."""
    middleware = _make_middleware(max_entries=2)

    _ = await middleware._coordination.get_flight_lock("q1", "m", "")
    _ = await middleware._coordination.get_flight_lock("q2", "m", "")
    _ = await middleware._coordination.get_flight_lock("q1", "m", "")
    _ = await middleware._coordination.get_flight_lock("q3", "m", "")

    assert len(middleware._coordination._flight_locks) == 2
    assert ("q1", "m", "") in middleware._coordination._flight_locks
    assert ("q2", "m", "") not in middleware._coordination._flight_locks
    assert ("q3", "m", "") in middleware._coordination._flight_locks


@pytest.mark.asyncio
async def test_get_flight_lock_skips_eviction_for_locked_entries() -> None:
    """Do not evict entries that are currently coordinating active requests."""
    middleware = _make_middleware(max_entries=2)

    held = await middleware._coordination.get_flight_lock("q1", "m", "")
    await held._lock.acquire()
    try:
        _ = await middleware._coordination.get_flight_lock("q2", "m", "")
        _ = await middleware._coordination.get_flight_lock("q3", "m", "")
    finally:
        held._lock.release()

    assert len(middleware._coordination._flight_locks) == 2
    assert ("q1", "m", "") in middleware._coordination._flight_locks
    assert ("q2", "m", "") not in middleware._coordination._flight_locks
    assert ("q3", "m", "") in middleware._coordination._flight_locks


@pytest.mark.asyncio
async def test_get_flight_lock_hard_cap_uncoordinated_when_registry_full() -> None:
    """New keys use an ephemeral lock when every registry slot is held."""
    middleware = _make_middleware(max_entries=1)

    held = await middleware._coordination.get_flight_lock("q1", "m", "")
    await held._lock.acquire()
    try:
        with patch("semanticcache.middleware.core.coordination._logger.warning") as (
            mock_warning
        ):
            ephemeral = await middleware._coordination.get_flight_lock("q2", "m", "")
        assert len(middleware._coordination._flight_locks) == 1
        assert ephemeral._lock is not held._lock
        assert ("q2", "m", "") not in middleware._coordination._flight_locks
        mock_warning.assert_called_once()
    finally:
        held._lock.release()


@pytest.mark.asyncio
async def test_flight_lock_removed_from_registry_after_exit() -> None:
    """Registry entry is removed once the flight context manager exits."""
    middleware = _make_middleware(max_entries=4)

    flight = await middleware._coordination.get_flight_lock("q1", "m", "")
    assert ("q1", "m", "") in middleware._coordination._flight_locks

    async with flight:
        assert ("q1", "m", "") in middleware._coordination._flight_locks

    assert ("q1", "m", "") not in middleware._coordination._flight_locks


@pytest.mark.asyncio
async def test_flight_lock_registry_does_not_accumulate_stale_entries() -> None:
    """Completed flights are removed so the registry stays small."""
    middleware = _make_middleware(max_entries=4)

    for i in range(10):
        flight = await middleware._coordination.get_flight_lock(f"q{i}", "m", "")
        async with flight:
            pass

    assert len(middleware._coordination._flight_locks) == 0


@pytest.mark.asyncio
async def test_flight_lock_shared_lock_not_removed_while_second_waiter_holds() -> None:
    """Registry entry persists while a second waiter still holds the lock."""
    import asyncio

    middleware = _make_middleware(max_entries=4)

    # First caller: get the flight lock and acquire it.
    flight1 = await middleware._coordination.get_flight_lock("q1", "m", "")
    await flight1._lock.acquire()

    # Second caller: gets the same registered lock (same key).
    flight2 = await middleware._coordination.get_flight_lock("q1", "m", "")
    assert flight1._lock is flight2._lock

    # First caller releases; lock transitions to flight2's acquire.
    acquire_task = asyncio.create_task(flight2._lock.acquire())
    flight1._lock.release()
    await acquire_task

    # Key should still be registered because flight2 is holding the lock.
    assert ("q1", "m", "") in middleware._coordination._flight_locks

    # Clean up.
    flight2._lock.release()
    async with middleware._coordination._flight_lock_registry:
        middleware._coordination._flight_locks.pop(("q1", "m", ""), None)
