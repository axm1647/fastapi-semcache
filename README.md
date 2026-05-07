# fastapi-semcache

Drop-in semantic caching for FastAPI APIs and LLM endpoints.

`fastapi-semcache` adds semantic response caching with minimal refactoring, using pgvector for similarity search and optional Redis for faster response lookups. It works as FastAPI middleware today and can also run as a reverse proxy in front of an upstream API or LLM service.

The PyPI distribution and GitHub repository are **`fastapi-semcache`**. The import package remains **`semanticcache`**.

## Why fastapi-semcache?

`fastapi-semcache` is built for Python teams who want semantic caching without rewriting their app around a larger framework.

It is designed to plug into FastAPI with minimal refactoring, while still giving you direct control over embeddings, similarity thresholds, vector storage, and cache behavior. The default setup keeps things simple: find the highest-similarity match, apply a threshold, and return a cached response only when it is safe to do so.

It supports FastAPI middleware as a first-class integration path and can also run as a reverse proxy in front of an upstream API or LLM service. Planned support for Django and Flask will extend the same integration model to other Python web stacks.

## Install

```bash
pip install fastapi-semcache
```

**Custom embedders:** subclass `BaseEmbedder` from `semanticcache.embedders` and pass it to `SemanticCache(embedder=...)` to skip the optional embedding extras. See [docs/embedders.md](docs/embedders.md).

Optional extras:

- `embed-huggingface` / `embed-huggingface-cpu`: Sentence Transformers with **CPU** PyTorch.
- `embed-huggingface-gpu`: Sentence Transformers with a CUDA-enabled PyTorch install.
- `embed-openai`: OpenAI embeddings (`openai`, `tiktoken`).

### CPU

```bash
pip install "fastapi-semcache[embed-huggingface-cpu]"
# or: pip install "fastapi-semcache[embed-huggingface]"
```

### GPU

Pick a CUDA version that matches your system from [PyTorch Get Started](https://pytorch.org/get-started/locally/), then install with that index so pip selects CUDA wheels.

```bash
pip install "fastapi-semcache[embed-huggingface-gpu]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

### OpenAI embeddings

Install the OpenAI extra so `embedder_type="openai"` works (pulls `openai` and `tiktoken`). Set `OPENAI_API_KEY` in your environment.

```bash
pip install "fastapi-semcache[embed-openai]"
```

## FastAPI middleware

Add `SemanticCacheMiddleware` to your app and reuse one `SemanticCache` instance for all requests. Configure Postgres, Redis, and the embedder with **`SEMANTIC_CACHE_*`** environment variables (see `.env.example`). By default only **`POST`** requests are intercepted; the middleware derives cache-key text from JSON bodies using `query`, `prompt`, `input`, or chat-style `messages` (see `default_extract_query` in `semanticcache.middleware`). Successful responses whose body parses as a **JSON object** are candidates for storage, and cache hits replay the original HTTP status and response metadata.

Redis is optional. If **`SEMANTIC_CACHE_REDIS_URI`** is empty (or whitespace), the cache runs in Postgres-only mode: semantic lookup and response storage still work via pgvector, but Redis TTL-based payload caching is disabled.

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
    # cache_scope / tenant_id); see docs/cache-tuning.md. Misses run your handler;
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

Use **`extract_scope`** (optional) when you need custom tenant or user routing; otherwise, with **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE`** left at its default **`true`**, the middleware uses **`X-Semantic-Cache-Scope`** and JSON **`cache_scope`** / **`tenant_id`** (numeric **`tenant_id`** is accepted). Treat header or body scope as trusted only if your gateway or app sets it from authenticated identity; otherwise clients can spoof another tenant id. Set **`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`** only for single-tenant apps or isolated cache storage. Scope rules in middleware match **`SemanticCache.settings`** when **`cache`** is a **`SemanticCache`** instance ( **`cache_settings`** still drives circuit breaker and flight-lock options). **`resolve_cache_scope`** matches the same rules for direct **`SemanticCache`** use.

See **`docs/cache-tuning.md`** for upgrade notes on **`scope_key`** and Redis key layout.

For **`create_semantic_cache_proxy_app`**, pass **`extract_query=...`** (and other middleware options) as keyword arguments; they are forwarded to `SemanticCacheMiddleware`.

Other advanced options (`path_prefix`, HTTP 429 circuit breaker via `cache_settings`, `enabled=False`) are documented on **`SemanticCacheMiddleware`** in `semanticcache.middleware.fastapi`. On shutdown, call `await cache.close()` from a lifespan handler if you want pools closed cleanly.

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

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

See `create_semantic_cache_proxy_app` in `semanticcache.proxy` for timeout, TLS verification, `httpx_client_kwargs`, and middleware options such as `path_prefix` and `extract_query`.

## Streaming and chunked responses

Today the middleware **buffers the full downstream response** before sending it to the client. That applies even when your route returns a streaming-style response (for example token streaming); the bytes are collected first, then returned as one response. Cached hits are served as ordinary JSON bodies. The reverse proxy uses httpx’s full response body, not a streamed upstream read.

**Chunked pass-through and streaming-friendly caching are planned** so SSE and similar flows can deliver early bytes while still integrating with semantic caching where feasible.

## Current features

- **Huggingface embeddings** via Sentence Transformers (`embedder_type="huggingface"`).
- **OpenAI embeddings** via the official async client (`embedder_type="openai"`; install
  `embed-openai` and set `OPENAI_API_KEY`). Use
  `OpenAIEmbedder(..., send_dimensions_to_api=False)` when the model has a fixed
  output size and the API must not get a `dimensions` field.
- **PostgreSQL + pgvector** for semantic similarity lookup. The library creates a
  dedicated cache table per embedder configuration (derived from model id and vector
  dimension) on first use, so you are not tied to a single hard-coded vector width.
- **Optional Redis** for response caching (keys include an embedder-specific prefix
  so separate models do not collide). If Redis is not configured, responses are read
  from Postgres only.

- **FastAPI middleware** for in-app semantic caching.
- **Reverse proxy mode** via `create_semantic_cache_proxy_app()`.

## Future support

- **Chunked / streaming responses** for the middleware (and related proxy behavior): pass-through streaming instead of full buffering; see [Streaming and chunked responses](#streaming-and-chunked-responses).
- **Django** and **Flask** middleware for in-app semantic caching (not yet shipped; same role as the FastAPI middleware).

Embeddings from the following providers are planned:

- **Ollama** (HTTP embedding API against a configurable base URL, so the server can run locally or on another host).
- **Cohere**
- **Voyage**

## Requirements

Python 3.12+.

## Links

- Repository: [fastapi-semcache](https://github.com/axm1647/fastapi-semcache)

## License

Apache-2.0. See `LICENSE`.
