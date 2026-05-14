![PyPI - Version](https://img.shields.io/pypi/v/fastapi-semcache)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastapi-semcache)
![License](https://img.shields.io/github/license/axm1647/fastapi-semcache?color=green)

# fastapi-semcache

Ultra-lightweight semantic caching middleware for FastAPI APIs and LLM endpoints.

`fastapi-semcache` adds semantic response caching as a thin async middleware layer. Vector similarity search runs inside Postgres via pgvector. Python never owns the heavy computation. It works as FastAPI middleware today and can also run as a reverse proxy in front of an upstream API or LLM service.

The PyPI distribution and GitHub repository are **`fastapi-semcache`**. The import package remains **`semanticcache`** (**`fastapi_semcache`** is available as an alias).

## Why fastapi-semcache?

`fastapi-semcache` is an ultra-lightweight middleware with a minimal, honest dependency footprint. The core install adds four packages to your project: `fastapi`, `pydantic-settings`, `psycopg` (libpq C bindings), and `httpx`. No vector database service to run, no ML runtime in the hot path, no framework to build around.

**Python is the glue, not the bottleneck.** In the request hot path, Python parses the JSON body, dispatches async I/O, and coordinates the result. That is all. Every computationally heavy operation is offloaded:

| What | Where it runs |
|------|--------------|
| Cosine / ANN vector similarity | Postgres + pgvector (C, indexed) |
| Embedding generation | Your provider's API (I/O, not CPU) |
| Response blob storage and retrieval | Postgres rows or Redis (C clients) |
| HTTP proxying | `httpx.AsyncClient` (async I/O) |

Because all meaningful work is either I/O-bound (GIL released) or executing inside a C extension, Python is never the ceiling even under high concurrency with a single `uvicorn` worker. The likely bottlenecks under load are your Postgres connection pool and your embedder API, not this middleware.

It plugs into FastAPI with minimal refactoring, while giving you direct control over embeddings, similarity thresholds, vector storage, and cache behavior. The default setup keeps things simple: find the highest-similarity match, apply a threshold, and return a cached response only when it is safe to do so.

It supports FastAPI middleware as a first-class integration path and can also run as a reverse proxy in front of an upstream API or LLM service. Planned support for Django and Flask will extend the same integration model to other Python web stacks.

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

**Custom embedders:** subclass `BaseEmbedder` from `semanticcache.embedders` and pass it to `SemanticCache(embedder=...)` to skip the optional embedding extras. See [docs/embedders.md](docs/embedders.md).

Optional extras:

- `redis`: Async Redis client (`redis>=7.4.0`) for TTL-backed response blobs when **`SEMANTIC_CACHE_REDIS_URI`** is set. Core installs omit it so Postgres-only deployments avoid pulling Redis.
- `embed-huggingface`: Sentence Transformers and PyTorch. Default PyPI wheels are **CPU**; for CUDA, install with PyTorch's `--extra-index-url` ([below](#hugging-face--sentence-transformers)).
- `embed-openai`: OpenAI embeddings (`openai`, `tiktoken`).
- `embed-voyage`: Voyage AI embeddings (`voyageai`, `aiohttp`).
- `embed-ollama`: Ollama embeddings via the OpenAI-compatible HTTP API (`openai` only).

Dependency notes:

- Core `fastapi-semcache` has no LangChain dependency.
- Core does **not** include the `redis` PyPI package; use **`pip install "fastapi-semcache[redis]"`** whenever you configure a non-empty Redis URI (otherwise the first Redis use raises `ImportError` with an install hint).
- Optional extras only add their listed packages (`redis`, `sentence-transformers`/`torch`, `openai`/`tiktoken`, `voyageai`/`aiohttp`, or `openai` alone for `embed-ollama`).

### Hugging Face / Sentence Transformers

```bash
pip install "fastapi-semcache[embed-huggingface]"
```

That pulls CPU PyTorch from PyPI. For **GPU (CUDA)**, use the same extra but pass PyTorch's wheel index so pip resolves CUDA builds. Pick a CUDA version that matches your system from [PyTorch Get Started](https://pytorch.org/get-started/locally/):

```bash
pip install "fastapi-semcache[embed-huggingface]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

### OpenAI embeddings

Install the OpenAI extra so `embedder_type="openai"` works (pulls `openai` and `tiktoken`). Set `OPENAI_API_KEY` in your environment.

```bash
pip install "fastapi-semcache[embed-openai]"
```

### Voyage embeddings

Install the Voyage extra so `embedder_type="voyage"` works (pulls `voyageai` and `aiohttp`). Set **`VOYAGE_API_KEY`** or **`SEMANTIC_CACHE_VOYAGE_API_KEY`**. Optional **`SEMANTIC_CACHE_VOYAGE_EMBEDDING_MODEL`** and **`SEMANTIC_CACHE_VOYAGE_EMBEDDING_DIMENSIONS`** default to **`voyage-4`** and **`1024`** when unset (they must match your chosen model and pgvector column width). Set **`SEMANTIC_CACHE_VOYAGE_INPUT_TYPE`** to **`query`** or **`document`** when you want Voyage’s input-type hint on each request.

```bash
pip install "fastapi-semcache[embed-voyage]"
```

### Ollama embeddings

Install the Ollama extra so `embedder_type="ollama"` works (pulls `openai` only). Set **`SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL`** and **`SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS`** to match the embedding model you run (dimensions must match pgvector). Optionally set **`SEMANTIC_CACHE_OLLAMA_BASE_URL`** (default `http://127.0.0.1:11434/v1`) and **`OLLAMA_API_KEY`** or **`SEMANTIC_CACHE_OLLAMA_API_KEY`** when your server uses auth.

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

> **Security: cache scope and cross-tenant isolation**
>
> By default (`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=true`), the middleware reads the
> cache partition key from client-controlled sources: the `X-Semantic-Cache-Scope`
> header and the `cache_scope` / `tenant_id` JSON body fields. **Any client can
> forge these values to read another tenant's cached responses or write responses
> into another tenant's cache partition.**
>
> This default is safe only for single-tenant apps (consider setting
> `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false` to remove the scope requirement
> entirely) or when a trusted edge/gateway overwrites those fields from verified
> identity before requests reach your app.
>
> **For multi-tenant APIs exposed directly to clients, always supply a server-side
> `extract_scope`:**
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

**`SEMANTIC_CACHE_PG_URI`** is required. Set it to your PostgreSQL connection string (e.g. `postgresql://user:pass@localhost:5432/semanticcache`). The library creates its cache table automatically on first use.

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
    # middleware can build the cache key (see default_extract_query). By default a
    # tenant scope is also required (header X-Semantic-Cache-Scope or JSON
    # cache_scope / tenant_id); those values are client-controlled unless you replace
    # extract_scope — unsuitable for multi-tenant production without a trusted edge
    # or server-side scope (see docs/cache-tuning.md). Misses run your handler;
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

Use **`extract_scope`** (optional) when you need custom tenant or user routing; otherwise, with **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE`** left at its default **`true`**, the middleware uses **`default_extract_scope_from_request_context`**, which reads **`X-Semantic-Cache-Scope`** and JSON **`cache_scope`** / **`tenant_id`** (numeric **`tenant_id`** is accepted). That default is appropriate only for single-tenant deployments or when a trusted gateway overwrites those fields from authenticated identity; otherwise clients can spoof another tenant id and probe or pollute another partition. For multi-tenant APIs exposed to clients, pass **`extract_scope`** that derives scope from server-side identity (see **`trusted_extract_scope_from_server_side`** in **`semanticcache.middleware.core.extractors`** after auth middleware sets **`request.state`**). Set **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`** only for single-tenant apps or isolated cache storage. Scope rules in middleware match **`SemanticCache.settings`** when **`cache`** is a **`SemanticCache`** instance ( **`cache_settings`** still drives circuit breaker and flight-lock options). **`resolve_cache_scope`** matches the same rules for direct **`SemanticCache`** use.

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

The reverse proxy mode is optional: it forwards traffic to an upstream base URL while using the same semantic cache middleware. Use it when you want a standalone hop in front of another service rather than importing routes into your FastAPI app.

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

This repository includes a small ASGI app at `app/main.py` (import `app` for uvicorn). Set **`SEMANTIC_CACHE_PROXY_UPSTREAM`** to the backend base URL; the default is `http://127.0.0.1:11434`. For semantic caching in front of a single trusted upstream, set **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`** unless you forward a tenant header or JSON scope from clients.

If your upstream requires an `Authorization` header (for example OpenAI-compatible APIs), set **`SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS=true`** or the middleware will bypass cache reads and writes for those requests.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

See `create_semantic_cache_proxy_app` in `semanticcache.proxy` for timeout, TLS verification, `httpx_client_kwargs`, and middleware options such as `path_prefix` and `extract_query`.

## Streaming and chunked responses

By default, `fastapi-semcache` uses a **buffered** response mode: the middleware buffers the full downstream response before sending it to the client and before writing to the cache.

### Miss delivery: `response_mode`

Set `response_mode` on `CacheSettings` to control how cache-miss responses are delivered:

- **`"buffered"` (default)** -- full response is buffered before the client receives anything; cache write completes before the response is returned.
- **`"tee"`** -- chunks are forwarded to the client as they arrive; the cache write runs in a background task after the stream completes. This gives lower time-to-first-byte for streaming upstreams (for example token streaming or SSE) while still accumulating the full body for storage.

In both modes the cache stores only fully assembled JSON object responses. The tee path respects the same store rules as the buffered path (headers, validation, and size limits).

### Hit delivery: `hit_response_mode`

Set `hit_response_mode` on `CacheSettings` to control how cache-hit responses are delivered:

- **`"single"` (default)** -- the cached body is returned as a single HTTP response.
- **`"stream"`** -- the cached body is emitted as ASGI body chunks, matching the framing of a streaming miss response. When `response_mode="tee"` and `hit_response_mode` is not explicitly set, it defaults to `"stream"` automatically so hit and miss delivery are symmetric.

Use `hit_stream_chunk_size` (env `SEMANTIC_CACHE_HIT_STREAM_CHUNK_SIZE`, default `0`) to split hit responses into multiple chunks of at most that many bytes. `0` sends the full body as a single chunk, which is sufficient for most clients; positive values are useful for clients that measure time-to-first-byte or process tokens incrementally.

### Reverse proxy note

For `create_semantic_cache_proxy_app`, upstream responses are fetched via `httpx.AsyncClient` using a buffered body, but `response_mode` and `hit_response_mode` control delivery to clients at the ASGI layer in the same way as the middleware.

## Current features

- **Huggingface embeddings** via Sentence Transformers (`embedder_type="huggingface"`).
- **OpenAI embeddings** via the official async client (`embedder_type="openai"`; install
  `embed-openai` and set `OPENAI_API_KEY`). Use
  `OpenAIEmbedder(..., send_dimensions_to_api=False)` when the model has a fixed
  output size and the API must not get a `dimensions` field.
- **Voyage AI embeddings** via aiohttp and the Voyage REST API (`embedder_type="voyage"`;
  install `embed-voyage` and set a Voyage API key). Defaults match **`VoyageEmbedder`**
  in code (`voyage-4`, 1024 dimensions) when model and dimensions are not set in env.
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

Embeddings from the following providers are planned:

- **Cohere**

## Requirements

Python 3.12+.

## Links

- Repository: [fastapi-semcache](https://github.com/axm1647/fastapi-semcache)

## License

Apache-2.0. See `LICENSE`.
