![PyPI - Version](https://img.shields.io/pypi/v/fastapi-semcache)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastapi-semcache)
![License](https://img.shields.io/github/license/axm1647/fastapi-semcache?color=green)

# fastapi-semcache

Semantic caching middleware for FastAPI APIs and LLM endpoints.

`fastapi-semcache` adds a thin async caching layer around your FastAPI app.
It embeds incoming requests, searches for similar cached responses in Postgres
with pgvector, and falls through to your handler on a miss. It can also run as a
reverse proxy in front of an upstream API or LLM service.

The PyPI distribution and GitHub repository are **`fastapi-semcache`**. The import package remains **`semanticcache`** (**`fastapi_semcache`** is available as an alias).

## Why fastapi-semcache?

`fastapi-semcache` is meant for projects that already use FastAPI and Postgres
and want semantic response caching without adding a separate vector database.
The core install adds `starlette` and `psycopg` (libpq C
bindings). Your app should already depend on `fastapi` when you use a
`FastAPI()` instance; the middleware itself is Starlette/ASGI middleware.
Reverse proxy mode installs `fastapi` and `aiohttp` via the optional `proxy`
extra.

In the request hot path, Python parses the JSON body, dispatches async I/O, and
coordinates the result. The heavier work happens elsewhere:

| What | Where it runs |
|------|--------------|
| Cosine / ANN vector similarity | Postgres + pgvector (C, indexed) |
| Embedding generation | Your provider's API (I/O, not CPU) |
| Response blob storage and retrieval | Postgres rows or Redis (C clients) |
| HTTP proxying | `aiohttp.ClientSession` (async I/O, optional `proxy` extra) |

Under load, the first things to watch are usually your Postgres connection pool
and your embedding provider, not Python CPU time in the middleware.

The middleware keeps the main choices explicit: embedder, similarity threshold,
vector storage, Redis usage, tenant scope, and cache behavior. By default it
looks for the closest match, applies the configured threshold, and returns a
cached response only when the match passes.

FastAPI middleware is the primary integration path. Reverse proxy mode is useful
when you want a standalone caching layer in front of an existing service.
Planned Django and Flask support will use the same general model.

## When not to use

Semantic caching works best when a meaningful fraction of queries share similar intent within your cache window. It is not a good fit for every endpoint:

- **Mostly unique queries**: creative writing, one-off analysis, ad-hoc exploration. Low hit rates mean you pay embedding cost on every request for little return.
- **Highly personalised responses**: where the same question requires a different answer per user. Scope partitioning helps, but if every partition is unique you gain nothing.
- **Constantly changing data**: real-time prices, live feeds, rapidly-evolving documents. Cached responses go stale faster than they help.
- **Strict correctness requirements**: legal, medical, financial contexts where a semantically similar but not identical query may need a materially different answer.
- **Side-effectful POST endpoints**: returning a cached response silently skips the side effect. Exclude those paths with a custom `extract_query` that returns `None`.
- **Exact-match inputs**: if your prompts are deterministic and identical across users, a plain Redis key is simpler and cheaper than a vector search.

See [docs/when-not-to-use.md](docs/when-not-to-use.md) for a fuller treatment with alternatives.

## Install

```bash
pip install fastapi-semcache
```

If you already have an embedding service, subclass `BaseEmbedder` from
`semanticcache.embedders` and pass it to `SemanticCache(embedder=...)`. You do
not need any of the optional embedding extras in that setup. See
[docs/embedders.md](docs/embedders.md).

Optional extras:

- `proxy`: Reverse proxy app factory (`fastapi`, `aiohttp` for upstream HTTP).
- `redis`: Async Redis client (`redis>=7.4.0`) for TTL-backed response blobs when **`SEMANTIC_CACHE_REDIS_URI`** is set. Core installs omit it so Postgres-only deployments avoid pulling Redis.
- `embed-huggingface`: Sentence Transformers and PyTorch. Default PyPI wheels are **CPU**; for CUDA, install with PyTorch's `--extra-index-url` ([below](#hugging-face--sentence-transformers)).
- `embed-openai`: OpenAI embeddings (`openai`, `tiktoken`).
- `embed-voyage`: Voyage AI embeddings (`voyageai`, `aiohttp`).
- `embed-cohere`: Cohere embeddings (`cohere`).
- `embed-ollama`: Ollama embeddings via the OpenAI-compatible HTTP API (`openai` only).

Notes:

- Core `fastapi-semcache` has no LangChain dependency.
- Core does **not** include the `fastapi` PyPI package; declare **`fastapi`** in your own app when you use **`FastAPI()`**. The **`proxy`** extra installs **`fastapi`** for **`create_semantic_cache_proxy_app`**.
- Core does **not** include the `redis` PyPI package; use **`pip install "fastapi-semcache[redis]"`** whenever you configure a non-empty Redis URI (otherwise the first Redis use raises `ImportError` with an install hint).
- Optional extras only add their listed packages (`fastapi`/`aiohttp` for `proxy`, `redis`, `sentence-transformers`/`torch`, `openai`/`tiktoken`, `cohere`, `voyageai`/`aiohttp`, or `openai` alone for `embed-ollama`).

### Hugging Face / Sentence Transformers

This is mainly useful for local development and tests. Loading a model
in-process adds memory and compute overhead to each embed call. For production,
use a hosted backend (`openai`, `cohere`, `voyage`, `ollama`) or a custom `BaseEmbedder`
that calls your own embedding service. Instantiating `SBERTEmbedder` emits a
one-time `UserWarning`.

```bash
pip install "fastapi-semcache[embed-huggingface]"
```

That pulls CPU PyTorch from PyPI. For **GPU (CUDA)**, use the same extra and pass
PyTorch's wheel index so pip resolves CUDA builds. Pick a CUDA version that
matches your system from [PyTorch Get Started](https://pytorch.org/get-started/locally/):

```bash
pip install "fastapi-semcache[embed-huggingface]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

### OpenAI embeddings

Install the OpenAI extra to use `embedder_type="openai"`. It pulls `openai` and
`tiktoken`. Set `OPENAI_API_KEY` in your environment.

```bash
pip install "fastapi-semcache[embed-openai]"
```

### Voyage embeddings

Install the Voyage extra to use `embedder_type="voyage"`. It pulls `voyageai`
and `aiohttp`. Set **`VOYAGE_API_KEY`** or
**`SEMANTIC_CACHE_VOYAGE_API_KEY`**. Optional
**`SEMANTIC_CACHE_VOYAGE_EMBEDDING_MODEL`** and
**`SEMANTIC_CACHE_VOYAGE_EMBEDDING_DIMENSIONS`** default to **`voyage-4`** and
**`1024`** when unset. They must match your chosen model and pgvector column
width. Set **`SEMANTIC_CACHE_VOYAGE_INPUT_TYPE`** to **`query`** or
**`document`** when you want Voyage's input-type hint on each request.

```bash
pip install "fastapi-semcache[embed-voyage]"
```

### Cohere embeddings

Install the Cohere extra to use `embedder_type="cohere"`. It pulls `cohere`.
Set **`COHERE_API_KEY`** or **`SEMANTIC_CACHE_COHERE_API_KEY`**. Optional
**`SEMANTIC_CACHE_COHERE_EMBEDDING_MODEL`**, **`SEMANTIC_CACHE_COHERE_EMBEDDING_DIMENSIONS`**,
and **`SEMANTIC_CACHE_COHERE_INPUT_TYPE`** default to **`embed-v4.0`**, **`1536`**,
and **`search_document`** when unset.

```bash
pip install "fastapi-semcache[embed-cohere]"
```

### Ollama embeddings

Install the Ollama extra to use `embedder_type="ollama"`. It pulls `openai`
only because Ollama exposes an OpenAI-compatible embeddings endpoint. Set
**`SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL`** and
**`SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS`** to match the embedding model you
run. The dimensions must match pgvector. Optionally set
**`SEMANTIC_CACHE_OLLAMA_BASE_URL`** (default `http://127.0.0.1:11434/v1`) and
**`OLLAMA_API_KEY`** or **`SEMANTIC_CACHE_OLLAMA_API_KEY`** when your server uses
auth.

```bash
pip install "fastapi-semcache[embed-ollama]"
```

### Redis response cache

Install the Redis extra when **`SEMANTIC_CACHE_REDIS_URI`** (or constructor **`redis_uri`**) is non-empty so **`redis.asyncio`** is available.

```bash
pip install "fastapi-semcache[redis]"
```

You can combine extras, for example **`pip install "fastapi-semcache[redis,embed-openai]"`** or **`pip install "fastapi-semcache[redis,embed-voyage]"`**.

## FastAPI middleware

**fastapi-semcache** implements the ASGI directly rather than subclassing Starlette's **BaseHTTPMiddleware** class. [Here](https://starlette.dev/middleware/)

> **Security: cache scope and cross-tenant isolation**
>
> By default (`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`), the cache uses one
> shared bucket (single-tenant). No client-supplied scope is required.
>
> For **multi-tenant** APIs, set `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=true` and
> supply a server-side `extract_scope` that derives scope from authenticated
> identity. Do not rely on client-controlled `X-Semantic-Cache-Scope` or JSON
> `cache_scope` / `tenant_id`; clients can forge those values to read or pollute
> another tenant's partition.
>
> **Multi-tenant example (server-side scope):**
>
> ```python
> from starlette.requests import Request
> from semanticcache.middleware.core.extractors import trusted_extract_scope_from_server_side
>
> async def extract_scope(request: Request, body: bytes) -> str | None:
>     return await trusted_extract_scope_from_server_side(request)
>
> app.add_middleware(SemanticCacheMiddleware, cache=cache, extract_scope=extract_scope)
> # Starlette executes middleware in reverse addition order:
> # YourAuthMiddleware runs first and populates request.state,
> # then SemanticCacheMiddleware reads it.
> app.add_middleware(YourAuthMiddleware)
> ```
>
> `trusted_extract_scope_from_server_side` reads only `request.state`, which
> clients cannot forge. See [docs/cache-tuning.md](docs/cache-tuning.md) for
> details.

Add `SemanticCacheMiddleware` to your app and reuse one `SemanticCache` instance for all requests. Configure Postgres, Redis, and the embedder with **`SEMANTIC_CACHE_*`** environment variables (see `.env.example`). By default only **`POST`** requests are intercepted; the middleware derives cache-key text from JSON bodies using `query`, `prompt`, `input`, or chat-style `messages` (see `default_extract_query` in `semanticcache.middleware`). Successful responses whose body parses as a **JSON object** are candidates for storage, and cache hits replay the original HTTP status and response metadata.

**`SEMANTIC_CACHE_PG_URI`** is required. Set it to your PostgreSQL connection string (e.g. `postgresql://user:pass@localhost:5432/semanticcache`). By default (**`SEMANTIC_CACHE_PG_ENSURE_SCHEMA=true`**) the library creates a dedicated pgvector table on first use, scoped to the embedder's `cache_namespace` and vector dimension. Set **`SEMANTIC_CACHE_PG_ENSURE_SCHEMA=false`** when your team manages DDL externally (migrations, DBA) or the application database role lacks `CREATE` privileges. Each embedder configuration gets its own table: the pgvector index covers only the rows for that model, so it stays compact and ANN probes scan fewer candidates. Separate tables also isolate autovacuum scheduling so a high-write model does not contend with a quiet one, and you can reindex or drop one model's table without touching any other.

Redis is optional. If **`SEMANTIC_CACHE_REDIS_URI`** is empty (or whitespace), the cache runs in Postgres-only mode: semantic lookup and response storage still work via pgvector, but Redis TTL-based payload caching is disabled. If you **do** set a Redis URI, install **`fastapi-semcache[redis]`** (see [Redis response cache](#redis-response-cache)).

```python
from typing import Any

from fastapi import FastAPI

from semanticcache import SemanticCache, SemanticCacheMiddleware

app = FastAPI()
cache = SemanticCache()
app.add_middleware(SemanticCacheMiddleware, cache=cache)


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    # Clients should send JSON with prompt, query, input, or chat messages so the
    # middleware can build the cache key (see default_extract_query). By default no
    # tenant scope is required (single-tenant shared bucket). For multi-tenant APIs set
    # SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=true and server-side extract_scope (see
    # docs/cache-tuning.md). Misses run your handler;
    # hits short-circuit with a cached JSON body.
    return {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
```

Run with `uvicorn mymodule:app --host 0.0.0.0 --port 8000`.

### Custom cache key text (`extract_query`)

If your JSON body does not follow the usual `query` / `prompt` / `messages` patterns, pass an **async** callable as **`extract_query`**. It receives the Starlette **`Request`** and the **raw body bytes** (already buffered by the middleware). Return a **non-empty string** to embed and look up; return **`None`** to skip semantic caching for that request (the route still runs).

If **`extract_query`** or **`extract_model`** raises, the middleware logs the error (with stack trace) and forwards the request upstream **without** calling the cache (same outcome as returning **`None`** from **`extract_query`**, but the route still runs).

You can wrap **`default_extract_query`** and add fallbacks for your own fields, or replace it entirely.

```python
from fastapi import FastAPI, Request

from semanticcache import SemanticCache
from semanticcache.middleware import SemanticCacheMiddleware, default_extract_query

async def extract_query(request: Request, body: bytes) -> str | None:
    base = await default_extract_query(request, body)
    if base is not None:
        return base
    # Parse ``body`` for your schema; return None to bypass the cache.
    return None

app = FastAPI()
cache = SemanticCache()
app.add_middleware(
    SemanticCacheMiddleware,
    cache=cache,
    extract_query=extract_query,
)
```

Use **`extract_model`** when the cache key should also vary by model id from headers or JSON (same async `(request, body) -> str | None` idea). That model id is passed through to **`SemanticCache.get` / `put`**, which scope Postgres rows and Redis payload keys per model bucket as described in **`docs/cache-tuning.md`**.

Use **`extract_scope`** when you need tenant or user routing. With **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`** (default), scope is optional and all requests share one bucket. For multi-tenant isolation, set **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=true`** and pass **`extract_scope`** that derives scope from server-side identity (see **`trusted_extract_scope_from_server_side`** in **`semanticcache.middleware.core.extractors`** after auth middleware sets **`request.state`**). Do not use client headers or JSON scope fields alone in production multi-tenant setups. Scope rules in middleware match **`SemanticCache.settings`** when **`cache`** is a **`SemanticCache`** instance ( **`cache_settings`** still drives circuit breaker and flight-lock options). **`resolve_cache_scope`** matches the same rules for direct **`SemanticCache`** use.

See **`docs/cache-tuning.md`** for upgrade notes on **`scope_key`** and Redis key layout.

For **`create_semantic_cache_proxy_app`**, pass **`extract_query=...`** (and other middleware options) as keyword arguments; they are forwarded to `SemanticCacheMiddleware`.

Use **`validate_response`** when a route or provider has a strict response schema and you want to avoid storing malformed payloads. The callback receives a `ResponseValidationContext` with the request, raw request body, upstream response, parsed JSON object, model, and scope. Return `False` to return the upstream response normally but skip the cache write.

```python
from semanticcache import ResponseValidationContext


def validate_response(context: ResponseValidationContext) -> bool:
    if context.request.url.path == "/v1/chat/completions":
        return isinstance(context.payload.get("choices"), list)
    return True


app.add_middleware(
    SemanticCacheMiddleware,
    cache=cache,
    validate_response=validate_response,
)
```

Other advanced options (`path_prefix`, HTTP 429 circuit breaker via `cache_settings`, `enabled=False`) are documented on **`SemanticCacheMiddleware`** in `semanticcache.middleware.adapters.fastapi` (or via the public import `semanticcache.middleware`). On shutdown, call `await cache.close()` from a lifespan handler if you want pools closed cleanly. The 429 circuit breaker is per-process; in multi-worker deployments each worker tracks its own counter independently.

### Cache behavior and tuning

`SemanticCache` uses a two-stage retrieval pipeline:

- A **primary similarity threshold** (`SEMANTIC_CACHE_THRESHOLD`) and **top-k candidate limit** (`SEMANTIC_CACHE_TOP_K_CANDIDATES`) control which nearest neighbors are fetched from pgvector.
- An optional **rejection threshold** (`SEMANTIC_CACHE_REJECTION_THRESHOLD`) can then filter out borderline matches; if no candidate passes this second stage, the middleware returns a cache miss.
- **Dependency timeouts** let you fail fast when providers or storage are slow:
  `SEMANTIC_CACHE_EMBED_TIMEOUT_SECONDS` applies to embedder calls, and
  `SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS` applies to Postgres/Redis operations.
  On timeout, `SemanticCache` raises a timeout error, middleware logs it, and
  request handling continues in fail-open mode.
- **In-flight lock registry cap** bounds middleware memory used for concurrent
  miss coordination: `SEMANTIC_CACHE_MIDDLEWARE_FLIGHT_LOCK_MAX_ENTRIES`
  limits retained `(query, model, scope)` lock keys and evicts least-recently-used
  unlocked entries when needed.
- **Body size limit** (`SEMANTIC_CACHE_MAX_BODY_BYTES`, default `10_485_760` - 10 MB):
  the middleware skips semantic caching for any request body that exceeds this limit
  and forwards the request upstream unchanged. Set to `0` to disable the limit
  entirely. `DEFAULT_MAX_BODY_BYTES` is exported from `semanticcache` if you need the
  constant directly.

See `docs/cache-tuning.md` for concrete tuning tips and examples.

## Reverse proxy

The reverse proxy mode is optional: it forwards traffic to an upstream base URL while using the same semantic cache middleware. Install the **`proxy`** extra first (pulls **`fastapi`** and **`aiohttp`**):

```bash
pip install "fastapi-semcache[proxy]"
```

Use it when you want a standalone hop in front of another service rather than importing routes into your FastAPI app.

Minimal programmatic setup:

```python
from semanticcache import SemanticCache, create_semantic_cache_proxy_app

cache = SemanticCache()
app = create_semantic_cache_proxy_app(
    upstream="http://127.0.0.1:11434",
    cache=cache,
)
```

Run with `uvicorn mymodule:app --host 0.0.0.0 --port 8080`.

This repository includes a small ASGI app at `app/main.py` (import `app` for uvicorn). Set **`SEMANTIC_CACHE_PROXY_UPSTREAM`** to the backend base URL; the default is `http://127.0.0.1:11434`. Single-tenant scope is the default (`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`). For multi-tenant proxy deployments, set **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=true`** and supply server-side **`extract_scope`**.

If your upstream requires an `Authorization` header (for example OpenAI-compatible APIs), set **`SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS=true`** or the middleware will bypass cache reads and writes for those requests.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

See `create_semantic_cache_proxy_app` in `semanticcache.proxy` for timeout, TLS verification, `aiohttp_session_kwargs`, and middleware options such as `path_prefix` and `extract_query`.

## Streaming and chunked responses

By default, `fastapi-semcache` uses a **buffered** response mode: the middleware buffers the full downstream response before sending it to the client and before writing to the cache.

### Miss delivery: `response_mode`

Set `response_mode` on `CacheSettings` to control how cache-miss responses are delivered:

- **`"buffered"` (default)**: full response is buffered before the client receives anything; cache write completes before the response is returned.
- **`"tee"`**: chunks are forwarded to the client as they arrive; the cache write runs in a background task after the stream completes. This gives lower time-to-first-byte for streaming upstreams (for example token streaming or SSE) while still accumulating the full body for storage.

In both modes the cache stores only fully assembled JSON object responses. The tee path respects the same store rules as the buffered path (headers, validation, and size limits).

### Hit delivery: `hit_response_mode`

Set `hit_response_mode` on `CacheSettings` to control how cache-hit responses are delivered:

- **`"single"` (default)**: the cached body is returned as a single HTTP response.
- **`"stream"`**: the cached body is emitted as ASGI body chunks, matching the framing of a streaming miss response. When `response_mode="tee"` and `hit_response_mode` is not explicitly set, it defaults to `"stream"` automatically so hit and miss delivery are symmetric.

Use `hit_stream_chunk_size` (env `SEMANTIC_CACHE_HIT_STREAM_CHUNK_SIZE`, default `0`) to split hit responses into multiple chunks of at most that many bytes. `0` sends the full body as a single chunk, which is sufficient for most clients; positive values are useful for clients that measure time-to-first-byte or process tokens incrementally.

### Reverse proxy note

For `create_semantic_cache_proxy_app`, upstream responses are fetched via `aiohttp.ClientSession` using a buffered body, but `response_mode` and `hit_response_mode` control delivery to clients at the ASGI layer in the same way as the middleware.

## Current features

- **Huggingface embeddings** via Sentence Transformers (`embedder_type="huggingface"`).
- **OpenAI embeddings** via the official async client (`embedder_type="openai"`; install
  `embed-openai` and set `OPENAI_API_KEY`). Use
  `OpenAIEmbedder(..., send_dimensions_to_api=False)` when the model has a fixed
  output size and the API must not get a `dimensions` field.
- **Voyage AI embeddings** via aiohttp and the Voyage REST API (`embedder_type="voyage"`;
  install `embed-voyage` and set a Voyage API key). Defaults match **`VoyageEmbedder`**
  in code (`voyage-4`, 1024 dimensions) when model and dimensions are not set in env.
- **Cohere embeddings** via the official async client (`embedder_type="cohere"`;
  install `embed-cohere` and set a Cohere API key). Defaults match **`CohereEmbedder`**
  in code (`embed-v4.0`, 1536 dimensions, `search_document` input type) when
  model and dimensions are not set in env.
- **Ollama embeddings** via the OpenAI-compatible **`/v1/embeddings`** endpoint
  (`embedder_type="ollama"`; install `embed-ollama`). Model id and vector dimensions are
  required in settings so pgvector storage matches the running model.
- **PostgreSQL + pgvector** for semantic similarity lookup. The library creates a
  dedicated cache table per embedder configuration (derived from model id and vector
  dimension) on first use, so you are not tied to a single hard-coded vector width.
- **Optional Redis** for response caching (keys include an embedder-specific prefix
  so separate models do not collide). If Redis is not configured, responses are read
  from Postgres only.

- **FastAPI middleware** for in-app semantic caching.
- **Reverse proxy mode** via `create_semantic_cache_proxy_app()`.

## Future support

- **Django** and **Flask** middleware for in-app semantic caching (not yet shipped; same role as the FastAPI middleware).

## Requirements

Python 3.12+.

## Links

- Repository: [fastapi-semcache](https://github.com/axm1647/fastapi-semcache)

## License

Apache-2.0. See `LICENSE`.
