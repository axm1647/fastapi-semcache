"""Async PostgreSQL + pgvector storage for cache embeddings and payloads."""

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import re
from typing import LiteralString, Self, cast

from psycopg import sql
from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

from ...types import CacheEntry

_IDENTIFIER_SAFE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def _vector_literal(embedding: list[float]) -> str:
    """Format a Python embedding list for PostgreSQL ``vector`` casts.

    Args:
        embedding: Dense embedding matching ``VECTOR(d)`` column dimension.

    Returns:
        Bracketed comma-separated floats accepted by ``::vector``.
    """
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _validate_table_name(name: str) -> str:
    """Ensure ``name`` is a safe unqualified SQL identifier.

    Args:
        name: Proposed table name (only lower-case ``sc_`` + hex is expected).

    Returns:
        The same string when valid.

    Raises:
        ValueError: If ``name`` is not a safe identifier.
    """
    if not _IDENTIFIER_SAFE.match(name) or ".." in name:
        msg = f"invalid vector table name: {name!r}"
        raise ValueError(msg)
    return name


class AsyncPgVectorStore:
    """Insert and ANN-search rows in a pgvector table using cosine distance."""

    _embedding_dim: int
    _table_name: str
    _pool: AsyncConnectionPool
    _schema_lock: asyncio.Lock
    _schema_ready: bool

    def __init__(
        self,
        pg_uri: str,
        *,
        table_name: str,
        embedding_dim: int,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        """Configure an async connection pool for a ``VECTOR(dim)`` cache table.

        Args:
            pg_uri: PostgreSQL connection URI (pgvector extension required).
            table_name: Target table; created by ``ensure_schema`` if missing.
            embedding_dim: Expected embedding length; must match ``VECTOR(d)``.
            min_pool_size: Minimum connections kept in the pool.
            max_pool_size: Maximum concurrent connections.
        """
        self._embedding_dim = embedding_dim
        self._table_name = _validate_table_name(table_name)
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False
        self._pool = AsyncConnectionPool(
            conninfo=pg_uri,
            min_size=min_pool_size,
            max_size=max_pool_size,
            open=False,
        )

    async def open(self) -> None:
        """Open the async pool (required before queries)."""
        await self._pool.open()

    async def close(self) -> None:
        """Close all pooled connections."""
        await self._pool.close()

    async def __aenter__(self) -> Self:
        """Open the pool for use as an async context manager."""
        await self.open()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the pool when leaving the context."""
        await self.close()

    async def ensure_schema(self) -> None:
        """Create the cache table and HNSW index when they do not exist."""
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            if self._embedding_dim < 1 or self._embedding_dim > 16000:
                msg = "embedding_dim out of supported range for VECTOR()"
                raise ValueError(msg)
            dim_lit = sql.SQL(cast(LiteralString, str(self._embedding_dim)))
            tbl = sql.Identifier(self._table_name)
            create_table = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {tbl} (
                  id SERIAL PRIMARY KEY,
                  query_text TEXT NOT NULL,
                  query_embedding VECTOR({dim}),
                  response JSONB NOT NULL,
                  created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """).format(tbl=tbl, dim=dim_lit)
            idx_name = sql.Identifier(f"{self._table_name}_hnsw")
            create_idx = sql.SQL("""
                CREATE INDEX IF NOT EXISTS {idx}
                ON {tbl}
                USING hnsw (query_embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """).format(idx=idx_name, tbl=tbl)
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(create_table)
                    await cur.execute(create_idx)
            self._schema_ready = True

    def _ensure_dim(self, embedding: list[float]) -> None:
        if len(embedding) != self._embedding_dim:
            msg = (
                f"embedding length {len(embedding)} does not match "
                f"VECTOR({self._embedding_dim})"
            )
            raise ValueError(msg)

    async def upsert(
        self,
        query_text: str,
        embedding: list[float],
        response: dict[str, object],
    ) -> int:
        """Insert a cache row with optional embedding and JSON response.

        Args:
            query_text: Original query string stored for debugging or display.
            embedding: Vector matching the table ``VECTOR(d)`` dimension.
            response: JSON-serializable payload stored in ``response`` JSONB.

        Returns:
            Primary key ``id`` of the inserted row.

        Raises:
            ValueError: If ``embedding`` length does not match ``embedding_dim``.
        """
        self._ensure_dim(embedding)
        vec = _vector_literal(embedding)
        tbl = sql.Identifier(self._table_name)
        insert = sql.SQL("""
            INSERT INTO {tbl} (query_text, query_embedding, response)
            VALUES (%s, %s::vector, %s::jsonb)
            RETURNING id
            """).format(tbl=tbl)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                _ = await cur.execute(insert, (query_text, vec, Json(response)))
                row = await cur.fetchone()
                if row is None:
                    msg = "cache insert failed"
                    raise RuntimeError(msg)
                return int(row[0])

    async def similarity_search_top_k(
        self,
        query_embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[CacheEntry]:
        """Find up to ``limit`` nearest rows and apply a similarity gate.

        Uses ``<=>`` (cosine distance). Similarity is ``1 - distance``, comparable
        to cosine similarity for normalized vectors.

        Args:
            query_embedding: Query vector of length ``embedding_dim``.
            threshold: Minimum similarity in ``[0.0, 1.0]`` for a hit.
            limit: Maximum number of candidates to return (top-k).

        Returns:
            List of ``CacheEntry`` objects ordered from highest to lowest similarity,
            filtered to those at or above ``threshold``. The list is empty on miss.

        Raises:
            ValueError: If ``query_embedding`` length does not match ``embedding_dim``.
        """
        if limit <= 0:
            return []
        self._ensure_dim(query_embedding)
        vec = _vector_literal(query_embedding)
        tbl = sql.Identifier(self._table_name)
        stmt = sql.SQL("""
            SELECT id, query_text, response,
                   (1 - (query_embedding <=> %s::vector)) AS similarity
            FROM {tbl}
            WHERE query_embedding IS NOT NULL
            ORDER BY query_embedding <=> %s::vector
            LIMIT %s
            """).format(tbl=tbl)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                _ = await cur.execute(stmt, (vec, vec, limit))
                rows = await cur.fetchall()
                entries: list[CacheEntry] = []
                for row in rows:
                    rid, qtext, resp, similarity = row
                    sim = float(similarity)
                    if sim < threshold:
                        continue
                    if not isinstance(resp, dict):
                        msg = (
                            "cache response column must deserialize to a JSON object"
                        )
                        raise TypeError(msg)
                    entries.append(
                        CacheEntry(
                            id=int(rid),
                            query_text=str(qtext),
                            response=resp,
                            similarity=sim,
                        )
                    )
                return entries

    async def similarity_search(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> CacheEntry | None:
        """Find the nearest row by cosine distance and apply a similarity gate.

        This compatibility wrapper delegates to ``similarity_search_top_k`` with
        ``limit=1``.

        Args:
            query_embedding: Query vector of length ``embedding_dim``.
            threshold: Minimum similarity in ``[0.0, 1.0]`` for a hit.

        Returns:
            ``CacheEntry`` for the single nearest neighbor if similarity is at or
            above ``threshold``; otherwise ``None``.

        Raises:
            ValueError: If ``query_embedding`` length does not match ``embedding_dim``.
        """
        entries = await self.similarity_search_top_k(
            query_embedding=query_embedding,
            threshold=threshold,
            limit=1,
        )
        if not entries:
            return None
        return entries[0]
