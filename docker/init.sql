-- pgvector must exist before cache tables are created. SemanticCache creates one
-- table per embedder configuration (name derived from ``cache_namespace`` and
-- dimension) on first use via ``AsyncPgVectorStore.ensure_schema``.
CREATE EXTENSION IF NOT EXISTS vector;
