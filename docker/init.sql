CREATE EXTENSION vector;

CREATE TABLE cache_entries (
  id SERIAL PRIMARY KEY,
  query_text TEXT NOT NULL,
  query_embedding VECTOR(384),
  response JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index (384-dim MiniLM)
CREATE INDEX cache_hnsw ON cache_entries
USING hnsw (query_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
