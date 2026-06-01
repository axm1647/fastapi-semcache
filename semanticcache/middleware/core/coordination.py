"""Coordination helpers for middleware flight locks and 429 circuit state."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from types import TracebackType

from semanticcache.exceptions import FlightLockAcquisitionTimeoutError

_logger = logging.getLogger(__name__)


class _FlightLock:
    """Async context manager that acquires a lock and removes it from the registry on exit.

    Attributes:
        _lock: Underlying asyncio lock.
        _key: Registry key tuple for this flight.
        _registry: Shared flight-lock ordered dict.
        _registry_guard: Mutex protecting the registry.
    """

    __slots__ = (
        "_lock",
        "_key",
        "_registry",
        "_registry_guard",
        "_acquire_timeout_seconds",
        "_acquired",
    )

    def __init__(
        self,
        lock: asyncio.Lock,
        key: tuple[str, str | None, str],
        registry: OrderedDict[tuple[str, str | None, str], asyncio.Lock],
        registry_guard: asyncio.Lock,
        *,
        acquire_timeout_seconds: float | None,
    ) -> None:
        """Store the lock and its registry coordinates.

        Args:
            lock: Underlying asyncio lock for this flight key.
            key: Registry key tuple ``(query, model, scope_storage)``.
            registry: Shared flight-lock ordered dict to clean up from.
            registry_guard: Mutex that protects the registry dict.
            acquire_timeout_seconds: Max seconds to wait for ``lock.acquire()``;
                or ``None`` to wait indefinitely.
        """
        self._lock = lock
        self._key = key
        self._registry = registry
        self._registry_guard = registry_guard
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._acquired = False

    async def __aenter__(self) -> "_FlightLock":
        """Acquire the underlying lock, optionally bounded by a timeout."""
        if self._acquire_timeout_seconds is None:
            await self._lock.acquire()
        else:
            try:
                await asyncio.wait_for(
                    self._lock.acquire(),
                    timeout=self._acquire_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise FlightLockAcquisitionTimeoutError(
                    timeout_seconds=self._acquire_timeout_seconds,
                    key=self._key,
                ) from exc
        self._acquired = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the lock and remove the registry entry."""
        if not self._acquired:
            return
        self._acquired = False
        self._lock.release()
        async with self._registry_guard:
            registered = self._registry.get(self._key)
            if registered is self._lock and not self._lock.locked():
                self._registry.pop(self._key, None)


class MiddlewareCoordination:
    """Coordinate in-flight request locks and 429 circuit breaker state."""

    def __init__(
        self,
        *,
        flight_lock_max_entries: int,
        flight_lock_acquire_timeout_seconds: float | None,
        circuit_breaker_enabled: bool,
        circuit_breaker_limit: int,
        circuit_breaker_open_seconds: float,
    ) -> None:
        """Initialize coordination state for one middleware instance.

        Args:
            flight_lock_max_entries: Maximum retained lock keys.
            flight_lock_acquire_timeout_seconds: Max seconds a waiter may block on
                ``lock.acquire()``; ``None`` waits indefinitely.
            circuit_breaker_enabled: Whether 429 circuit logic is active.
            circuit_breaker_limit: Consecutive 429s required to open circuit.
            circuit_breaker_open_seconds: Circuit cooldown duration in seconds.
        """
        self._flight_lock_registry = asyncio.Lock()
        self._flight_locks: OrderedDict[tuple[str, str | None, str], asyncio.Lock] = (
            OrderedDict()
        )
        self._flight_lock_max_entries = max(1, flight_lock_max_entries)
        self._flight_lock_acquire_timeout_seconds = flight_lock_acquire_timeout_seconds

        self._circuit_breaker_enabled = circuit_breaker_enabled
        self._circuit_breaker_limit = circuit_breaker_limit
        self._circuit_breaker_open_seconds = circuit_breaker_open_seconds
        self._circuit_lock = asyncio.Lock()
        self._consecutive_429_count = 0
        self._circuit_open_until: float | None = None

    def _evict_unused_flight_locks(self) -> None:
        """Evict least-recently-used unlocked flight locks when over capacity."""
        while len(self._flight_locks) > self._flight_lock_max_entries:
            removed_any = False
            for key, lock in list(self._flight_locks.items()):
                if lock.locked():
                    continue
                self._flight_locks.pop(key, None)
                removed_any = True
                break
            if not removed_any:
                return

    async def get_flight_lock(
        self,
        query: str,
        model: str | None,
        scope_storage: str,
    ) -> _FlightLock:
        """Return context manager that serializes miss handling for one cache key.

        Acquiring the returned context manager locks the underlying
        ``asyncio.Lock``; releasing it removes the registry entry so the slot
        is available for new keys immediately after the flight completes.

        Args:
            query: Extracted cache key text.
            model: Optional model discriminator.
            scope_storage: Resolved storage scope string.

        Returns:
            ``_FlightLock`` for this ``(query, model, scope)`` tuple. When the
            registry is at capacity and every retained lock is held, returns a
            wrapper that is not registered; concurrent requests for the same key
            may then miss deduplication until capacity frees up.
        """
        key = (query, model, scope_storage)
        async with self._flight_lock_registry:
            lock = self._flight_locks.get(key)
            if lock is not None:
                self._flight_locks.move_to_end(key)
                return _FlightLock(
                    lock,
                    key,
                    self._flight_locks,
                    self._flight_lock_registry,
                    acquire_timeout_seconds=self._flight_lock_acquire_timeout_seconds,
                )
            lock = asyncio.Lock()
            self._flight_locks[key] = lock
            self._evict_unused_flight_locks()
            if key not in self._flight_locks:
                _logger.warning(
                    "Flight lock registry is full (%d distinct keys); this key was "
                    "evicted immediately because every retained lock was held. "
                    "Serving without registry coordination; concurrent identical keys "
                    "may duplicate upstream work until capacity frees.",
                    self._flight_lock_max_entries,
                )
            return _FlightLock(
                lock,
                key,
                self._flight_locks,
                self._flight_lock_registry,
                acquire_timeout_seconds=self._flight_lock_acquire_timeout_seconds,
            )

    async def upstream_blocked_by_circuit(self) -> bool:
        """Return True when 429 circuit is open and upstream must not be called.

        Returns:
            True if upstream call must be skipped.
        """
        if not self._circuit_breaker_enabled:
            return False
        async with self._circuit_lock:
            if self._circuit_open_until is None:
                return False
            now = time.monotonic()
            if now >= self._circuit_open_until:
                self._circuit_open_until = None
                return False
            return True

    async def record_upstream_status_for_circuit(self, status_code: int) -> None:
        """Update 429 circuit state from one upstream status code.

        Args:
            status_code: Upstream HTTP status code.
        """
        if not self._circuit_breaker_enabled:
            return
        async with self._circuit_lock:
            if status_code == 429:
                self._consecutive_429_count += 1
                if self._consecutive_429_count >= self._circuit_breaker_limit:
                    self._circuit_open_until = (
                        time.monotonic() + self._circuit_breaker_open_seconds
                    )
                    self._consecutive_429_count = 0
                    _logger.warning(
                        "Semantic cache 429 circuit breaker opened for %.2f seconds "
                        "after %i consecutive 429 response(s) from upstream",
                        self._circuit_breaker_open_seconds,
                        self._circuit_breaker_limit,
                    )
            else:
                self._consecutive_429_count = 0
