# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`SemanticCacheMiddleware`**: **`max_request_body_bytes`** and **`max_response_body_bytes`** (default **10 MiB** each, shared constant **`DEFAULT_MAX_BODY_BYTES`**) limit buffered request and upstream response size; **HTTP 413** and **HTTP 502** when exceeded.
- **Flight lock registry**: when every older retained in-flight lock is still held and a new distinct key is inserted, LRU eviction drops that key’s table entry immediately; **`MiddlewareCoordination`** emits a **critical** log so saturation and possible duplicate upstream work are visible.

### Fixed

- **`SemanticCacheMiddleware`** / **`response_from_cache_hit`**: similarity hits whose stored JSON omits the **`__semanticcache_record_v1__`** replay envelope (marker plus **`body`** and **`meta`**) are no longer answered as **HTTP 200** with no way to restore the original status code or headers. They are treated like other unreplayable hits (warning log, miss path, and Postgres or Redis eviction when **`SemanticCache`** supplies **`cache_entry_id`**).
- **`AsyncPgVectorStore.similarity_search_top_k`**: apply the similarity threshold in SQL before ``LIMIT`` so top-k retrieval considers only rows at or above the threshold (not the first k neighbors by distance).
- **`RedisResponseStore`** / **`SemanticCache`**: when **`store_timeout_seconds`** is set, configure **`socket_timeout`** and **`socket_connect_timeout`** on **`redis.asyncio.from_url`** to match so stalled Redis TCP or reads align with the asyncio store timeout instead of relying only on **`wait_for`**.
- **`SemanticCacheMiddleware`**: detect **`Set-Cookie`** via raw response headers before storing responses so multi-cookie or sparse header mappings cannot cache session-bearing responses.
- **`SemanticCacheMiddleware`**: decide at initialization whether ``cache.put`` accepts ``query_embedding`` (via ``inspect.signature`` on ``SemanticCache`` or duck-typed caches) instead of catching ``TypeError`` and matching the exception message when storing.

### Documentation

- **`docs/cache-tuning.md`**: request and response body size limits for **`SemanticCacheMiddleware`**; Stage 1 notes that Postgres applies the primary threshold before ``LIMIT``; saturated flight lock registry (LRU behavior when all older locks are held, deduplication gap, critical log); Redis **`from_url`** socket timeouts when **`SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS`** is set; duck-typed ``cache.put`` and ``query_embedding`` detection at middleware startup.
- **`docs/cache-tuning.md`**: unreplayable similarity hits when the replay marker or **`body`** / **`meta`** envelope is missing; **Scope key and Redis layout** replaces older **`scope_key`** migration wording with a short operational description (**`require_cache_scope`** and shared bucket behavior).

## [0.2.22] - 2026-05-10

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
