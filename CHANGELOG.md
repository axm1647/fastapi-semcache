# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [Unreleased]

### Added

- **`CacheSettings`**: raise UserWarning when **`CacheSettings.pg_uri`** equals the default dev credentials
- **`expires_at` column** (`AsyncPgVectorStore`): nullable `TIMESTAMPTZ` column added to every cache table. Controlled by **`SEMANTIC_CACHE_PG_TTL_DAYS`** (`CacheSettings.pg_ttl_days`, fractional float). When set, each upserted row receives `expires_at = NOW() + interval`; on conflict the deadline is refreshed. When unset (default), `expires_at` is `NULL` and rows never expire. Existing tables are migrated automatically via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` on first start. Row deletion is out of scope; schedule cleanup externally (e.g. `pg_cron`).
- **Symmetric streaming on cache hits** (`SemanticCacheMiddleware`): two new `CacheSettings` fields close the asymmetry where tee-mode misses streamed chunks to the client but hits returned a single `JSONResponse`.
  - **`hit_response_mode`** (`SEMANTIC_CACHE_HIT_RESPONSE_MODE`, `"single"` | `"stream"`): controls how cache-hit responses are delivered. `"stream"` emits the cached body as raw ASGI `http.response.body` chunks with no `content-length`, matching the framing of a tee miss. **When `response_mode="tee"` and this field is not explicitly set, it defaults to `"stream"` automatically** so hit and miss delivery are symmetric without any extra configuration. Set to `"single"` to revert to the previous `JSONResponse` path.
  - **`hit_stream_chunk_size`** (`SEMANTIC_CACHE_HIT_STREAM_CHUNK_SIZE`, default `0`): maximum byte length of each synthetic body chunk when `hit_response_mode="stream"`. `0` sends the full body as one chunk; a positive value splits it into sequential chunks of at most that size, useful for clients that measure time-to-first-byte or process tokens incrementally.
  - Security-sensitive headers (`set-cookie`, `authorization`, `www-authenticate`, `proxy-authenticate`) and `content-length` are always stripped from hit responses in stream mode.

## [0.3.1] - 2026-05-12

### Fixed

- **`_vector_literal`**: add a guard against non-finite embedding values and raises an error when values are non-finite.
- **`AsyncPgVectorStore.similarity_search_top_k`**: bind the query vector once via a CTE (`WITH q AS (SELECT %s::vector AS v)`) instead of three separate `%s::vector` parameters, reducing per-query wire payload ~3× for large embeddings.
- **`AsyncPgVectorStore.upsert`**: add `ON CONFLICT (query_text, model_key, scope_key) DO UPDATE` so concurrent cache misses for the same query converge to a single row rather than accumulating duplicates; returns the existing row `id` on conflict.
- **`AsyncPgVectorStore.ensure_schema`**: add a B-tree index on `(scope_key, model_key)` so multi-tenant similarity search prunes by tenant before ANN distance computation instead of scanning the full table; add a unique index on `(query_text, model_key, scope_key)` to back the `ON CONFLICT` clause in `upsert`.
- **`CacheSettings.upstream_timeout_seconds`** / **`SemanticCacheMiddleware`**: the upstream ASGI timeout now applies in **`response_mode='buffered'`** (via **`call_downstream`** with **`asyncio.wait_for`**) and on passthrough paths that invoke the same helper. Previously it only bounded the tee miss path, so a hung upstream in buffered mode could hold a worker indefinitely despite the setting. When the budget is exceeded the middleware cancels the downstream call, logs a warning, and returns **HTTP 504**. **`docs/cache-tuning.md`** and the setting field description now describe both modes.
- **`cache_record_from_response`** / **`response_from_cache_hit`**: strip `set-cookie`, `authorization`, `www-authenticate`, and `proxy-authenticate` headers before writing a cache record so session cookies and auth tokens are never persisted to Postgres or Redis and cannot be replayed to unrelated clients. The same filter is applied at replay time as defense-in-depth for any records stored before this fix.

## [0.3.0] - 2026-05-11

### Added

- **`CacheSettings.response_mode`**: **`buffered`** (default) or **`tee`**. Tee mode forwards downstream ASGI body chunks on cache misses while buffering for a post-response cache write (via **`SEMANTIC_CACHE_RESPONSE_MODE`** or **`SemanticCache.settings`** when the middleware uses cache settings for scope).
- **`SemanticCacheMiddleware`**: **`max_request_body_bytes`** and **`max_response_body_bytes`** (default **10 MiB** each, shared constant **`DEFAULT_MAX_BODY_BYTES`**) limit buffered request and upstream response size; **HTTP 413** and **HTTP 502** when exceeded.
- **Flight lock registry**: when every older retained in-flight lock is still held and a new distinct key is inserted, LRU eviction drops that key’s table entry immediately; **`MiddlewareCoordination`** emits a **critical** log so saturation and possible duplicate upstream work are visible.
- **Compatibility import**: **`fastapi_semcache`** imports no longer raise **`ImportError`** and instead re-imports from **`semanticcache/__init__.py`**

### Fixed

- **`SemanticCacheMiddleware`** / **`response_from_cache_hit`**: similarity hits whose stored JSON omits the **`__semanticcache_record_v1__`** replay envelope (marker plus **`body`** and **`meta`**) are no longer answered as **HTTP 200** with no way to restore the original status code or headers. They are treated like other unreplayable hits (warning log, miss path, and Postgres or Redis eviction when **`SemanticCache`** supplies **`cache_entry_id`**).
- **`AsyncPgVectorStore.similarity_search_top_k`**: apply the similarity threshold in SQL before ``LIMIT`` so top-k retrieval considers only rows at or above the threshold (not the first k neighbors by distance).
- **`RedisResponseStore`** / **`SemanticCache`**: when **`store_timeout_seconds`** is set, configure **`socket_timeout`** and **`socket_connect_timeout`** on **`redis.asyncio.from_url`** to match so stalled Redis TCP or reads align with the asyncio store timeout instead of relying only on **`wait_for`**.
- **`SemanticCacheMiddleware`**: detect **`Set-Cookie`** via raw response headers before storing responses so multi-cookie or sparse header mappings cannot cache session-bearing responses.
- **`SemanticCacheMiddleware`**: decide at initialization whether ``cache.put`` accepts ``query_embedding`` (via ``inspect.signature`` on ``SemanticCache`` or duck-typed caches) instead of catching ``TypeError`` and matching the exception message when storing.
- **`SemanticCacheMiddleware`**: when both **`cache_settings`** and **`cache.settings`** are supplied and disagree on **`require_cache_scope`** or **`cache_authorized_requests`**, log a warning naming the field and both values so the split-source misconfiguration is visible at startup. The middleware still starts and applies the documented precedence.

### Documentation

- **`docs/cache-tuning.md`**: request and response body size limits for **`SemanticCacheMiddleware`**; Stage 1 notes that Postgres applies the primary threshold before ``LIMIT``; saturated flight lock registry (LRU behavior when all older locks are held, deduplication gap, critical log); Redis **`from_url`** socket timeouts when **`SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS`** is set; duck-typed ``cache.put`` and ``query_embedding`` detection at middleware startup.
- **`docs/cache-tuning.md`**: unreplayable similarity hits when the replay marker or **`body`** / **`meta`** envelope is missing; **Scope key and Redis layout** replaces older **`scope_key`** migration wording with a short operational description (**`require_cache_scope`** and shared bucket behavior).
- **`docs/cache-tuning.md`**: settings alignment now notes that **`cache_settings`** also governs **`cache_authorized_requests`** and that the middleware emits a startup warning when split sources disagree.

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
