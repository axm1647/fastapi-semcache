"""Minimal Redis JSON blob store for cached HTTP-like responses."""

# pyright: reportAny=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import redis.asyncio as redis_async


def _require_redis():
    """Import redis.asyncio or raise with install hint."""
    try:
        import redis.asyncio as redis_asyncio
    except ImportError as exc:
        msg: str = (
            "RedisResponseStore requires optional dependencies. pip install "
            "'fastapi-semcache[redis]'"
        )
        raise ImportError(msg) from exc
    return redis_asyncio


class RedisResponseStore:
    """GET/SET JSON-serializable dicts under a stable key prefix with optional TTL."""

    _key_prefix: str
    _default_ttl_seconds: int | None
    _redis_uri: str
    _socket_timeout_seconds: float | None
    _socket_connect_timeout_seconds: float | None
    _client: redis_async.Redis | None

    def __init__(
        self,
        redis_uri: str,
        *,
        key_prefix: str = "semanticcache:resp:",
        default_ttl_seconds: int | None = None,
        socket_timeout_seconds: float | None = None,
        socket_connect_timeout_seconds: float | None = None,
    ) -> None:
        """Attach to Redis using a shared-async client.

        Args:
            redis_uri: Connection URL for ``redis.asyncio.from_url``.
            key_prefix: Prepended to every logical ``key`` from ``get``/``put``.
            default_ttl_seconds: Used when ``put`` is called without ``ttl_seconds``.
                If both are omitted, keys persist until explicitly overwritten.
            socket_timeout_seconds: Redis ``socket_timeout`` in seconds for read/write
                operations; omit for ``redis.asyncio`` defaults.
            socket_connect_timeout_seconds: Redis ``socket_connect_timeout`` in seconds
                for the TCP connect phase; omit for ``redis.asyncio`` defaults.
        """
        self._key_prefix = key_prefix
        self._default_ttl_seconds = default_ttl_seconds
        self._redis_uri = redis_uri
        self._socket_timeout_seconds = socket_timeout_seconds
        self._socket_connect_timeout_seconds = socket_connect_timeout_seconds
        self._client = None

    def _client_or_create(self) -> redis_async.Redis:
        redis_asyncio = _require_redis()

        if self._client is None:
            kwargs: dict[str, object] = {"decode_responses": True}
            if self._socket_timeout_seconds is not None:
                kwargs["socket_timeout"] = self._socket_timeout_seconds
            if self._socket_connect_timeout_seconds is not None:
                kwargs["socket_connect_timeout"] = (
                    self._socket_connect_timeout_seconds
                )
            self._client = redis_asyncio.from_url(self._redis_uri, **kwargs)
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    async def open(self) -> None:
        """Ensure the Redis client is constructed (optional convenience)."""
        _ = self._client_or_create()

    async def close(self) -> None:
        """Release the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        """Open client for async context manager usage."""
        await self.open()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close Redis when leaving the context."""
        await self.close()

    async def get(self, key: str) -> dict[str, object] | None:
        """Load a JSON object stored under the prefixed key.

        Args:
            key: Logical key without the configured prefix.

        Returns:
            Parsed dict if present; ``None`` on cache miss.
        """
        r = self._client_or_create()
        raw = await r.get(self._full_key(key))
        if raw is None:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            msg = "Redis payload must be a JSON object at the top level"
            raise TypeError(msg)
        return data

    async def put(
        self,
        key: str,
        value: dict[str, object],
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Serialize ``value`` to JSON and store it under the prefixed key.

        Args:
            key: Logical key without the configured prefix.
            value: JSON-serializable object (stored as a JSON object).
            ttl_seconds: Expiry in seconds. Falls back to ``default_ttl_seconds``;
                if still unset, the key has no TTL.

        Raises:
            TypeError: If stored JSON round-trip does not yield a ``dict``.
        """
        r = self._client_or_create()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        fk = self._full_key(key)
        if ttl is not None and ttl > 0:
            await r.set(fk, payload, ex=int(ttl))
        else:
            await r.set(fk, payload)

    async def delete(self, key: str) -> None:
        """Remove the stored JSON value for a logical key when present.

        Args:
            key: Logical key without the configured prefix.
        """
        r = self._client_or_create()
        await r.delete(self._full_key(key))
