"""Coordination helpers for middleware flight locks and 429 circuit state."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

_logger = logging.getLogger(__name__)


class MiddlewareCoordination:
    """Coordinate in-flight request locks and 429 circuit breaker state."""

    def __init__(
        self,
        *,
        flight_lock_max_entries: int,
        circuit_breaker_enabled: bool,
        circuit_breaker_limit: int,
        circuit_breaker_open_seconds: float,
    ) -> None:
        """Initialize coordination state for one middleware instance.

        Args:
            flight_lock_max_entries: Maximum retained lock keys.
            circuit_breaker_enabled: Whether 429 circuit logic is active.
            circuit_breaker_limit: Consecutive 429s required to open circuit.
            circuit_breaker_open_seconds: Circuit cooldown duration in seconds.
        """
        self._flight_lock_registry = asyncio.Lock()
        self._flight_locks: OrderedDict[tuple[str, str | None, str], asyncio.Lock] = (
            OrderedDict()
        )
        self._flight_lock_max_entries = max(1, flight_lock_max_entries)

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
    ) -> asyncio.Lock:
        """Return lock that serializes miss handling for one cache key.

        Args:
            query: Extracted cache key text.
            model: Optional model discriminator.
            scope_storage: Resolved storage scope string.

        Returns:
            Async lock for this `(query, model, scope)` tuple. When the
            registry is at capacity and every retained lock is held, returns a
            lock that is not registered; concurrent requests for the same key
            may then miss deduplication until capacity frees up.
        """
        key = (query, model, scope_storage)
        async with self._flight_lock_registry:
            lock = self._flight_locks.get(key)
            if lock is not None:
                self._flight_locks.move_to_end(key)
                return lock
            lock = asyncio.Lock()
            self._flight_locks[key] = lock
            self._evict_unused_flight_locks()
            if key not in self._flight_locks:
                _logger.critical(
                    "Flight lock registry is full (%d distinct keys); this key was "
                    "evicted immediately because every retained lock was held. "
                    "Serving without registry coordination; concurrent identical keys "
                    "may duplicate upstream work until capacity frees.",
                    self._flight_lock_max_entries,
                )
            return lock

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
