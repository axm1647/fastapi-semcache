# fastapi-semcache

Semantic caching middleware and reverse proxy for APIs and LLMs, with embeddings, pgvector similarity search, and Redis-backed response caching.

The PyPI distribution and GitHub repository are **`fastapi-semcache`** (the import package remains **`semanticcache`**).

## Why fastapi-semcache?

This package is designed for direct integration into modern Python API stacks with minimal refactoring needed. It keeps the caching path simple and gives you explicit control over embeddings, vector search, and cache behavior.

It includes **FastAPI** middleware as a first-class integration path and can also run as a reverse proxy in front of an upstream API or LLM service. **Django** and **Flask** middleware are planned for a future release so you can hook semantic caching into those stacks the same way as FastAPI.

## FastAPI middleware

Add `SemanticCacheMiddleware` to your app and reuse one `SemanticCache` instance for all requests. Configure Postgres, Redis, and the embedder with **`SEMANTIC_CACHE_*`** environment variables (see `.env.example`). By default only **`POST`** requests are intercepted; the middleware derives cache-key text from JSON bodies using `query`, `prompt`, `input`, or chat-style `messages` (see `default_extract_query` in `semanticcache.middleware`). Successful responses whose body parses as a **JSON object** are candidates for storage.

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
    # middleware can build the cache key (see default_extract_query). Misses run your
    # handler; hits short-circuit with a cached JSON body.
    return {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
```

Run with `uvicorn mymodule:app --host 0.0.0.0 --port 8000`.

### Custom cache key text (`extract_query`)

If your JSON body does not follow the usual `query` / `prompt` / `messages` patterns, pass an **async** callable as **`extract_query`**. It receives the Starlette **`Request`** and the **raw body bytes** (already buffered by the middleware). Return a **non-empty string** to embed and look up; return **`None`** to skip semantic caching for that request (the route still runs).

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

Use **`extract_model`** when the cache key should also vary by model id from headers or JSON (same async `(request, body) -> str | None` idea). For **`create_semantic_cache_proxy_app`**, pass **`extract_query=...`** (and other middleware options) as keyword arguments; they are forwarded to `SemanticCacheMiddleware`.

Other advanced options (`path_prefix`, HTTP 429 circuit breaker via `cache_settings`, `enabled=False`) are documented on **`SemanticCacheMiddleware`** in `semanticcache.middleware.fastapi`. On shutdown, call `await cache.close()` from a lifespan handler if you want pools closed cleanly.

## What is implemented

- **Huggingface embeddings** via Sentence Transformers (`embedder_type="huggingface"`).
- **OpenAI embeddings** via the official async client (`embedder_type="openai"`; install
  `embed-openai` and set `OPENAI_API_KEY`). Use
  `OpenAIEmbedder(..., send_dimensions_to_api=False)` when the model has a fixed
  output size and the API must not get a `dimensions` field.
- **PostgreSQL + pgvector** for semantic similarity lookup. The library creates a
  dedicated cache table per embedder configuration (derived from model id and vector
  dimension) on first use, so you are not tied to a single hard-coded vector width.
- **Redis** for response caching (keys include an embedder-specific prefix so separate
  models do not collide).

- **FastAPI middleware** for in-app semantic caching.
- **Reverse proxy mode** via `create_semantic_cache_proxy_app()`.

## Streaming and chunked responses

Today the middleware **buffers the full downstream response** before sending it to the client. That applies even when your route returns a streaming-style response (for example token streaming); the bytes are collected first, then returned as one response. Cached hits are served as ordinary JSON bodies. The reverse proxy uses httpx’s full response body, not a streamed upstream read.

**Chunked pass-through and streaming-friendly caching are planned** so SSE and similar flows can deliver early bytes while still integrating with semantic caching where feasible.

## Future support

- **Chunked / streaming responses** for the middleware (and related proxy behavior): pass-through streaming instead of full buffering; see [Streaming and chunked responses](#streaming-and-chunked-responses).
- **Django** and **Flask** middleware for in-app semantic caching (not yet shipped; same role as the FastAPI middleware).

Embeddings from the following providers are planned:

- **Ollama** (HTTP embedding API against a configurable base URL, so the server can run locally or on another host).
- **Cohere**
- **Voyage**

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

This repository includes a small ASGI app at `app/main.py` (import `app` for uvicorn). Set **`SEMANTIC_CACHE_PROXY_UPSTREAM`** to the backend base URL; the default is `http://127.0.0.1:11434`.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

See `create_semantic_cache_proxy_app` in `semanticcache.proxy` for timeout, TLS verification, `httpx_client_kwargs`, and middleware options such as `path_prefix` and `extract_query`.

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

## Requirements

Python 3.12+.

## Links

- Repository: [fastapi-semcache](https://github.com/axm1647/fastapi-semcache)

## License

Apache-2.0. See `LICENSE`.
