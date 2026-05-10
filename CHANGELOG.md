# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`VoyageEmbedder`** (`semanticcache.embedders`): embeddings via `aiohttp` against `https://api.voyageai.com/v1/embeddings`; local input validation with `voyageai.Client.tokenize`. Install with **`fastapi-semcache[embed-voyage]`** (`voyageai`, `aiohttp`).
- **`get_embedder`** support when **`SEMANTIC_CACHE_EMBEDDER_TYPE=voyage`**, with **`CacheSettings`** fields **`voyage_embedding_model`**, **`voyage_embedding_dimensions`**, and **`voyage_input_type`** (defaults **`voyage-3`** / **`1024`** when unset).
- **`CacheResult.cache_entry_id`**: set on vector hits from **`SemanticCache.get`** for the matching Postgres row.
- **`SemanticCache.delete_entry_by_id`**: deletes that row (scoped by model and scope buckets) and the matching Redis response key when Redis is enabled.
- **`AsyncPgVectorStore.delete_by_id`** and **`RedisResponseStore.delete`** for targeted eviction.

### Fixed

- **`SemanticCacheMiddleware`**: similarity hits whose stored payload cannot be replayed (for example marked cache records with a non-object **`body`**) no longer fail silently. The middleware logs a warning and, when **`SemanticCache`** supplies **`cache_entry_id`**, removes the bad Postgres row and Redis key so the same corrupt entry does not win retrieval on every request.

### Documentation

- **`docs/embedders.md`**: Voyage embedder section (constructor, env vars, example).
- **`docs/cache-tuning.md`**: unreplayable similarity hits (logging and eviction behavior).

## [0.2.21] - 2026-05-09

### Added

- `trusted_extract_scope_from_server_side` in `semanticcache.middleware.core.extractors`: reads cache scope only from `request.state.cache_scope` or `request.state.tenant_id` for use after trusted auth or gateway middleware sets `request.state`.

### Changed

- Renamed `default_extract_scope` to `default_extract_scope_from_request_context` (same behavior: `X-Semantic-Cache-Scope` header and JSON `cache_scope` / `tenant_id`). Documented that this path trusts client-supplied values and is aimed at single-tenant or trusted-proxy setups.
- `SemanticCacheMiddleware` documentation now refers to the renamed default extractor and points multi-tenant deployments at a custom `extract_scope` or the new trusted helper.

### Removed

- **`default_extract_scope`** export from `semanticcache.middleware.core` (use **`default_extract_scope_from_request_context`** instead).

### Documentation

- README quick-start and `docs/cache-tuning.md`: clarified scope trust boundaries and added an example wrapping `trusted_extract_scope_from_server_side` for `extract_scope`.
