"""Async PostgreSQL + pgvector storage for cache embeddings and payloads."""

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Self

from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

from semanticcache.types import CacheEntry


def _vector_literal(embedding: list[float]) -> str:
    """Format a Python embedding list for PostgreSQL ``vector`` casts.

    Args:
        embedding: Dense embedding matching ``VECTOR(d)`` column dimension.

    Returns:
        Bracketed comma-separated floats accepted by ``::vector``.
    """
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


class AsyncPgVectorStore:
    """Insert and ANN-search rows in ``cache_entries`` using pgvector cosine ops."""

    _embedding_dim: int
    _pool: AsyncConnectionPool

    def __init__(
        self,
        pg_uri: str,
        *,
        embedding_dim: int = 384,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        """Configure an async connection pool for ``cache_entries``.

        Args:
            pg_uri: PostgreSQL connection URI (pgvector extension required).
            embedding_dim: Expected embedding length; must match ``VECTOR(d)``.
            min_pool_size: Minimum connections kept in the pool.
            max_pool_size: Maximum concurrent connections.
        """
        self._embedding_dim = embedding_dim
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
        insert = """
            INSERT INTO cache_entries (query_text, query_embedding, response)
            VALUES (%s, %s::vector, %s::jsonb)
            RETURNING id
            """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                _ = await cur.execute(insert, (query_text, vec, Json(response)))
                row = await cur.fetchone()
                if row is None:
                    msg = "cache_entries insert failed"
                    raise RuntimeError(msg)
                return int(row[0])

    async def similarity_search(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> CacheEntry | None:
        """Find the nearest row by cosine distance and apply a similarity gate.

        Uses ``<=>`` (cosine distance). Similarity is ``1 - distance``, comparable
        to cosine similarity for normalized vectors.

        Args:
            query_embedding: Query vector of length ``embedding_dim``.
            threshold: Minimum similarity in ``[0.0, 1.0]`` for a hit.

        Returns:
            ``CacheEntry`` for the single nearest neighbor if similarity is at or
            above ``threshold``; otherwise ``None``.

        Raises:
            ValueError: If ``query_embedding`` length does not match ``embedding_dim``.
        """
        self._ensure_dim(query_embedding)
        vec = _vector_literal(query_embedding)
        stmt = """
            SELECT id, query_text, response,
                   (1 - (query_embedding <=> %s::vector)) AS similarity
            FROM cache_entries
            WHERE query_embedding IS NOT NULL
            ORDER BY query_embedding <=> %s::vector
            LIMIT 1
            """
        async with self._pool.connection() as conn:
            
            async with conn.cursor() as cur:
                _ = await cur.execute(stmt, (vec, vec))
                row = await cur.fetchone()
                if row is None:
                    return None
                rid, qtext, resp, similarity = row
                sim = float(similarity)
                if sim < threshold:
                    return None
                if not isinstance(resp, dict):
                    msg = "cache_entries.response must deserialize to a JSON object"
                    raise TypeError(msg)
                return CacheEntry(
                    id=int(rid),
                    query_text=str(qtext),
                    response=resp,
                    similarity=sim,
                )
