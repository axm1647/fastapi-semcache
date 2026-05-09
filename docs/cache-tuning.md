## Cache behavior and similarity tuning

`SemanticCache` uses a two-stage retrieval model so you can trade off recall and precision without changing application code.

On cache misses handled by `SemanticCacheMiddleware`, the embedding computed during
`get()` is now reused by `put()` for the same request. This removes one embedder
call per miss, reducing latency and external embedding API cost.

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

**Trust boundary:** Header and JSON scope values are only safe isolation boundaries when your deployment sets them (for example from verified JWT claims at the edge) or overwrites untrusted client fields before they reach this middleware. Otherwise a client can pick another tenant id and probe for cache hits; always derive scope from authenticated identity in multi-tenant systems.

**Settings alignment:** `SemanticCacheMiddleware` applies `require_cache_scope` and the gate for “missing scope” using **`SemanticCache.settings`** when the `cache` argument is a real `SemanticCache` instance. `cache_settings` still controls circuit breaker and flight-lock limits. Avoid passing a different `require_cache_scope` only via `cache_settings` while using a `SemanticCache` with conflicting settings.

Integer **`tenant_id`** (JSON number) is accepted and normalized to a string for storage keys.

**Single-tenant exception:** Set `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false` only when one customer owns the process **and** dedicated cache storage, or when you intentionally share one global cache bucket.

Middleware in-flight lock keys also include the resolved **scope** string so concurrent misses for different tenants are not serialized together.

### Upgrading and operations

Adding **`scope_key`** changes how Postgres rows match and reshapes Redis key segments (extra scope bucket hash before the model segment). After upgrading, expect older Redis entries not to be reused until they expire; pgvector rows created before migration keep `scope_key = ''`, which only participates in lookups when `require_cache_scope` is false (legacy shared bucket). Turning **`require_cache_scope`** on in an existing deployment effectively starts fresh tenant partitions until new responses are cached.

### Stage 1: nearest-neighbor search (top-k)

The first stage embeds the query and runs a pgvector similarity search:

- **`SEMANTIC_CACHE_THRESHOLD`** (`CacheSettings.threshold`):
  - Primary similarity gate in \[0.0, 1.0].
  - Candidates below this value are discarded by the vector store.
- **`SEMANTIC_CACHE_TOP_K_CANDIDATES`** (`CacheSettings.top_k_candidates`):
  - Maximum number of nearest neighbors returned from pgvector after applying the primary threshold.
  - Defaults to `1` for single-hit behavior.

After this stage you get up to `top_k_candidates` `CacheEntry` rows ordered from highest to lowest similarity, all with `similarity >= threshold`.

### Stage 2: optional rejection threshold

The second stage can apply a stricter similarity cutoff on the in-memory candidates:

- **`SEMANTIC_CACHE_REJECTION_THRESHOLD`** (`CacheSettings.rejection_threshold`):
  - When unset (`None` or empty env var), behavior matches the original single-threshold model: the best remaining candidate is accepted.
  - When set, it must be **greater than or equal to** `SEMANTIC_CACHE_THRESHOLD`; otherwise settings validation fails (a lower value would make the second stage unable to reject anything that passed the primary gate).
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
- When `SEMANTIC_CACHE_REJECTION_THRESHOLD` is set, it must satisfy `rejection_threshold >= threshold`.

## Timeout tuning

Slow embedder providers or storage dependencies can increase request latency and
tie up worker capacity. `SemanticCache` supports fail-fast timeout controls:

- **`SEMANTIC_CACHE_EMBED_TIMEOUT_SECONDS`**
  (`CacheSettings.embed_timeout_seconds`):
  timeout budget for embedder calls used by `get()` and `put()`.
- **`SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS`**
  (`CacheSettings.store_timeout_seconds`):
  timeout budget for Postgres and Redis operations, including initial pool open
  and schema checks.

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

Default is `4096`. If all tracked locks are currently held, temporary growth
above the cap is possible until one becomes idle.

