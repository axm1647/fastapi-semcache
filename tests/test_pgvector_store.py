"""Focused tests for pgvector store query-time tuning behavior."""

from __future__ import annotations

import pytest

from semanticcache.stores.vector.pgvector import AsyncPgVectorStore


class _FakeCursor:
    """Capture executed statements and return fixed result rows."""

    def __init__(
        self,
        *,
        rows: list[tuple[int, str, dict[str, object], float]],
    ) -> None:
        self.rows = rows
        self.execute_calls: list[tuple[object, object | None]] = []

    async def execute(
        self,
        query: object,
        params: object | None = None,
    ) -> None:
        self.execute_calls.append((query, params))

    async def fetchall(self) -> list[tuple[int, str, dict[str, object], float]]:
        return self.rows


class _FakeCursorContext:
    """Expose a fake cursor through an async context manager."""

    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    async def __aenter__(self) -> _FakeCursor:
        return self._cursor

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        _ = exc_type, exc, tb


class _FakeConnection:
    """Expose a fake cursor factory matching psycopg connection usage."""

    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursorContext:
        return _FakeCursorContext(self._cursor)


class _FakeConnectionContext:
    """Expose a fake connection through an async context manager."""

    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        _ = exc_type, exc, tb


class _FakePool:
    """Expose a fake pool connection context used by the store."""

    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def connection(self) -> _FakeConnectionContext:
        return _FakeConnectionContext(self._connection)


@pytest.mark.asyncio
async def test_similarity_search_uses_store_default_hnsw_ef_search() -> None:
    """Store-level ef_search default is applied transaction-locally."""
    cursor = _FakeCursor(rows=[(1, "q", {"answer": 1}, 0.93)])
    store = AsyncPgVectorStore(
        "postgresql://mock/mock",
        table_name="sc_test_default",
        embedding_dim=4,
        hnsw_ef_search=80,
    )
    store._pool = _FakePool(_FakeConnection(cursor))

    entries = await store.similarity_search_top_k(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        threshold=0.8,
        limit=1,
    )

    assert len(entries) == 1
    assert cursor.execute_calls[0] == (
        "SELECT set_config('hnsw.ef_search', %s, true)",
        ("80",),
    )


@pytest.mark.asyncio
async def test_similarity_search_override_wins_over_store_default() -> None:
    """Per-call ef_search override takes precedence over the configured default."""
    cursor = _FakeCursor(rows=[(1, "q", {"answer": 1}, 0.93)])
    store = AsyncPgVectorStore(
        "postgresql://mock/mock",
        table_name="sc_test_override",
        embedding_dim=4,
        hnsw_ef_search=80,
    )
    store._pool = _FakePool(_FakeConnection(cursor))

    await store.similarity_search_top_k(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        threshold=0.8,
        limit=1,
        ef_search=120,
    )

    assert cursor.execute_calls[0] == (
        "SELECT set_config('hnsw.ef_search', %s, true)",
        ("120",),
    )
