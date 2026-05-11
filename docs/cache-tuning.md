## Cache behavior and similarity tuning

`SemanticCache` uses a two-stage retrieval model so you can trade off recall and precision without changing application code.

On cache misses handled by `SemanticCacheMiddleware`, the embedding computed during
`get()` is now reused by `put()` for the same request. This removes one embedder
call per miss, reducing latency and external embedding API cost.

When you pass a duck-typed `cache` (not a real `SemanticCache`), implement
`put(..., *, query_embedding=...)` when you want that reuse. At startup the
middleware inspects `cache.put` and only omits `query_embedding` if the signature
does not accept it, so storage does not rely on fragile runtime `TypeError`
string checks.

### Model-scoped storage

`SemanticCache.get` and `SemanticCache.put` accept an optional **`model`** string (for example an LLM id from JSON or a header). The value is normalized (stripped; `None` or blank becomes the **default bucket**, `model_key=""`). Lookup and writes are scoped:

- **Postgres:** Rows carry a `model_key` column; ANN search only considers rows for that bucket.
- **Redis:** Response keys include short hashes of the scope and model buckets plus the row id so payloads never collide across tenants or models for the same embedder row. Enable Redis by setting **`SEMANTIC_CACHE_REDIS_URI`** and install the **`redis`** extra: `pip install "fastapi-semcache[redis]"` (provides `redis>=7.4.0`; omitted from the core wheel).

Pass the **same** `model` on `get` and `put` for a given upstream route.

### Tenant and namespace scope (isolation)

Semantic matches are keyed by middleware lookup text and model. In middleware mode,
lookup text includes HTTP method, normalized path, model value, and extracted
semantic query, then tenant scope is applied separately. This avoids accidental
cross-endpoint reuse for semantically similar prompts.

**Default (scope required):** `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE` (`CacheSettings.require_cache_scope`) is **true**. Then:

- `SemanticCache.get(..., scope=...)` / `put(..., scope=...)` require a **non-empty** normalized scope string. Missing scope yields a cache **miss** and **skips** `put` (no cross-tenant writes).
- Pass the **same** `scope` on `get` and `put` as you use for tenant, org, or user partition (opaque string from your auth layer).
- Use **`resolve_cache_scope`** to mirror middleware rules in custom integrations.

**Middleware:** When `require_cache_scope` is true and you omit **`extract_scope`**, the middleware uses **`default_extract_scope_from_request_context`**, which reads `X-Semantic-Cache-Scope` and JSON fields `cache_scope` or `tenant_id`. That path trusts client headers and body; use it only for single-tenant setups or when a trusted proxy sets those fields. For multi-tenant production, pass **`extract_scope`** (`(request, body) -> str | None`) that resolves scope from authenticated identity. A concrete helper is **`trusted_extract_scope_from_server_side`** (`semanticcache.middleware.core.extractors`), which reads only **`request.state.cache_scope`** or **`request.state.tenant_id`** after your auth middleware populates them:

```python
from semanticcache.middleware.core.extractors import trusted_extract_scope_from_server_side

async def extract_scope(request, body: bytes) -> str | None:
    return await trusted_extract_scope_from_server_side(request)
```

For privacy and HTTP cache-safety alignment, middleware also skips cache writes when upstream responds with `Cache-Control: no-store`, `Cache-Control: private`, or any `Set-Cookie` header.

Middleware also bypasses cache reads and writes for requests that include an `Authorization` header unless you explicitly opt in with `SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS=true` (`CacheSettings.cache_authorized_requests`). This default reduces accidental reuse of per-user responses across authenticated callers.
This is especially important for reverse-proxy deployments because upstream APIs often require `Authorization`; without this setting those requests always miss and never write cache entries.

### Request and response body size limits

`SemanticCacheMiddleware` buffers the full request body and the full downstream response. To cap memory use and reduce abuse from huge payloads, use **`max_request_body_bytes`** and **`max_response_body_bytes`**. Each defaults to **`DEFAULT_MAX_BODY_BYTES`** (10 MiB). When a client request exceeds the request cap, the middleware answers with **HTTP 413** before the route runs. When the upstream response would exceed the response cap, the client receives **HTTP 502** (the handler may still have run; the middleware does not forward an oversized body). Set either argument to **`None`** to disable that limit (not recommended in untrusted or high-concurrency production setups). The same options are accepted by **`create_semantic_cache_proxy_app`** via keyword arguments.

**Response delivery on miss:** `SEMANTIC_CACHE_RESPONSE_MODE` (`CacheSettings.response_mode`) is **`buffered`** by default (full body buffered before the client sees the response). Set to **`tee`** to stream chunks to the client on cache misses while still accumulating the body for a post-stream cache write (when within size limits and validation passes). When the middleware uses **`SemanticCache.settings`** for the scope gate, **`response_mode`** is read from that same object; otherwise it comes from the middleware **`cache_settings`** source (see middleware constructor docs).

### Response shape validation

Middleware stores successful responses only when the body parses as a JSON object.
For provider-specific APIs, add `validate_response` to reject malformed or
mismatched objects before they can become cache entries. The validator can be
sync or async and receives `ResponseValidationContext` with the route request,
raw request body, upstream response, parsed payload, model, and scope.

```python
from semanticcache import ResponseValidationContext, SemanticCacheMiddleware


def validate_response(context: ResponseValidationContext) -> bool:
    if context.request.url.path == "/v1/chat/completions":
        return (
            context.model == "gpt-5.4-mini"
            and isinstance(context.payload.get("choices"), list)
        )
    return True


app.add_middleware(
    SemanticCacheMiddleware,
    cache=cache,
    validate_response=validate_response,
)
```

Returning `False`, or raising from the validator, skips the cache write while
still returning the upstream response to the caller.

### Unreplayable similarity hits

When ANN search returns a row but the stored JSON is not a replayable response (for
example the replay marker is set but `body` is not a JSON object, or the marker and
`body` / `meta` envelope are missing),
`SemanticCacheMiddleware` logs a warning, treats the lookup as a miss, and calls
downstream. If the cache backend is `SemanticCache` and `CacheResult` includes
`cache_entry_id`, the middleware also deletes that Postgres row (and the matching
Redis key when Redis is enabled) so one bad row cannot force repeated misses.

**Trust boundary:** Header and JSON scope values are only safe isolation boundaries when your deployment sets them (for example from verified JWT claims at the edge) or overwrites untrusted client fields before they reach this middleware. Otherwise a client can pick another tenant id and probe for cache hits; always derive scope from authenticated identity in multi-tenant systems.

**Settings alignment:** `SemanticCacheMiddleware` applies `require_cache_scope` and the gate for “missing scope” using **`SemanticCache.settings`** when the `cache` argument is a real `SemanticCache` instance. `cache_settings` still controls circuit breaker, flight-lock limits, and the `cache_authorized_requests` gate. Avoid passing a different `require_cache_scope` only via `cache_settings` while using a `SemanticCache` with conflicting settings. When both sources are supplied and disagree on `require_cache_scope` or `cache_authorized_requests`, the middleware logs a warning at startup naming the field and both values so the misconfiguration is visible without breaking the app.

Integer **`tenant_id`** (JSON number) is accepted and normalized to a string for storage keys.

**Single-tenant exception:** Set `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false` only when one customer owns the process **and** dedicated cache storage, or when you intentionally share one global cache bucket.

Middleware in-flight lock keys also include the resolved **scope** string so concurrent misses for different tenants are not serialized together.

### Scope key and Redis layout

**`scope_key`** affects Postgres matching and Redis key segments (an extra scope bucket hash appears before the model segment). Rows with `scope_key = ''` are looked up only when `require_cache_scope` is false (one shared bucket). When **`require_cache_scope`** is true, each normalized scope string is its own partition.

### Stage 1: nearest-neighbor search (top-k)

The first stage embeds the query and runs a pgvector similarity search:

- **`SEMANTIC_CACHE_THRESHOLD`** (`CacheSettings.threshold`):
  - Primary similarity gate in \[0.0, 1.0].
  - Candidates below this value are discarded in Postgres before ``LIMIT`` so the
    top-k cap applies only among rows that meet the threshold.
- **`SEMANTIC_CACHE_TOP_K_CANDIDATES`** (`CacheSettings.top_k_candidates`):
  - Maximum number of nearest neighbors returned from pgvector after applying the primary threshold.
  - Defaults to `1` for single-hit behavior.

After this stage you get up to `top_k_candidates` `CacheEntry` rows ordered from highest to lowest similarity, all with `similarity >= threshold`.

### Stage 2: optional rejection threshold

The second stage can apply a stricter similarity cutoff on the in-memory candidates:

- **`SEMANTIC_CACHE_REJECTION_THRESHOLD`** (`CacheSettings.rejection_threshold`):
  - When unset (`None` or empty env var), behavior matches the original single-threshold model: the best remaining candidate is accepted.
  - When set, it must be **greater than or equal to** `SEMANTIC_CACHE_THRESHOLD`; otherwise settings validation fails (a lower value would make the second stage unable to reject anything that passed the primary gate).
  - If it **equals** `SEMANTIC_CACHE_THRESHOLD`, validation still succeeds, but you get a **warning at startup**: the second stage cannot filter out any candidate that passed the primary gate (same cutoff). Use a strictly higher rejection threshold if you want stage 2 to matter, or leave rejection unset.
  - When set, the cache scans the candidates in order and selects the **first** entry whose `similarity >= rejection_threshold`.
  - If **no** candidate passes this stricter bar, the cache returns a **miss** (`is_hit=False`).

This is useful when you want to:

- Keep `SEMANTIC_CACHE_THRESHOLD` lower (for example `0.80`) to allow more candidates into the first stage.
- Enforce a higher bar (for example `0.90`) for actually serving a cached response.

### Example configurations

- **Strict, precision-first cache:**

  - `SEMANTIC_CACHE_THRESHOLD=0.90`
  - `SEMANTIC_CACHE_TOP_K_CANDIDATES=1`
  - `SEMANTIC_CACHE_REJECTION_THRESHOLD=` (unset)

  Only very similar neighbors are considered, and the single best one is accepted if it passes `0.90`.

- **More recall with a second-stage guard:**

  - `SEMANTIC_CACHE_THRESHOLD=0.80`
  - `SEMANTIC_CACHE_TOP_K_CANDIDATES=5`
  - `SEMANTIC_CACHE_REJECTION_THRESHOLD=0.90`

  Up to five neighbors with similarity at least `0.80` are fetched; the cache only serves a hit if at least one of them has similarity `>= 0.90`, otherwise it falls back to a miss.

### Notes

- If `SEMANTIC_CACHE_TOP_K_CANDIDATES` is less than `1`, it is treated as `1` internally.
- All thresholds are clamped to the inclusive range \[0.0, 1.0] by `CacheSettings`.
- When `SEMANTIC_CACHE_REJECTION_THRESHOLD` is set, it must satisfy `rejection_threshold >= threshold`. Equality issues a warning because the second stage has no effect (see above).

## Timeout tuning

Slow embedder providers or storage dependencies can increase request latency and
tie up worker capacity. `SemanticCache` supports fail-fast timeout controls:

- **`SEMANTIC_CACHE_EMBED_TIMEOUT_SECONDS`**
  (`CacheSettings.embed_timeout_seconds`):
  timeout budget for embedder calls used by `get()` and `put()`.
- **`SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS`**
  (`CacheSettings.store_timeout_seconds`):
  timeout budget for Postgres and Redis operations, including initial pool open
  and schema checks. When this value is set (non-null), the Redis response store
  also passes it as ``socket_timeout`` and ``socket_connect_timeout`` to
  ``redis.asyncio.from_url`` so stalled TCP or Redis reads cannot block the
  event loop beyond that budget at the socket layer. If you disable the store
  timeout (null/empty env), Redis uses library defaults for those socket options.

When these timeouts are exceeded, the cache raises a timeout exception with
operation metadata, emits a warning log entry, and increments an in-process
operation timeout counter (`SemanticCache.timeout_counts`) for observability.
Middleware continues to fail open, so requests still execute against upstream
handlers.

## Middleware in-flight lock registry

`SemanticCacheMiddleware` keeps an in-memory lock table to serialize concurrent
cache misses for the same `(method + normalized path + model + semantic query, scope)` key. To prevent unbounded growth in
long-lived processes with high key cardinality, configure:

- **`SEMANTIC_CACHE_MIDDLEWARE_FLIGHT_LOCK_MAX_ENTRIES`**
  (`CacheSettings.middleware_flight_lock_max_entries`):
  maximum number of distinct in-flight lock keys retained. When the limit is
  exceeded, the middleware evicts least-recently-used **unlocked** lock entries.
  Locks currently coordinating active requests are never evicted.

Default is `4096`. **Saturated registry:** when every older retained lock is
still held and a new distinct key is inserted, LRU eviction drops that new key’s
table entry immediately (the new lock is the first unlocked slot in traversal
order). The caller still holds the same lock object, but it is no longer tracked,
so concurrent identical keys are not deduplicated until capacity frees. A
critical-level log is emitted when this happens.

