"""Derive stable Postgres table and Redis key prefixes per embedder configuration."""

from __future__ import annotations

import hashlib


def embedding_storage_ids(namespace: str, embedding_dim: int) -> tuple[str, str]:
    """Return SQL table name and Redis key prefix for an embedder namespace.

    Table names are short hashes so several models or dimensions can coexist without
    collisions or identifier length issues.

    Args:
        namespace: Value from ``BaseEmbedder.cache_namespace``.
        embedding_dim: Vector length for this configuration.

    Returns:
        Tuple of ``(postgres_table_name, redis_key_prefix)``.
    """
    payload = f"{namespace}\0{embedding_dim}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:20]
    table = f"sc_{digest}"
    redis_prefix = f"sc:{digest}"
    return table, redis_prefix
