# fastapi-semcache

[![PyPI - Version](https://img.shields.io/pypi/v/fastapi-semcache)](https://pypi.org/project/fastapi-semcache/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastapi-semcache)](https://pypi.org/project/fastapi-semcache/)
[![License](https://img.shields.io/github/license/axm1647/fastapi-semcache?color=green)](https://github.com/axm1647/fastapi-semcache/blob/main/LICENSE)

Ultra-lightweight semantic caching middleware for FastAPI APIs and LLM endpoints.

`fastapi-semcache` adds semantic response caching as a thin async middleware layer. Vector similarity search runs inside Postgres via **pgvector**. Python never owns the heavy computation. It works as FastAPI middleware today and can also run as a reverse proxy in front of an upstream API or LLM service.

---

## How it works

When a request arrives, the middleware:

1. Extracts the semantic query text from the request body (`query`, `prompt`, `input`, `messages` or your own extractor callable).
2. Embeds the query using a configurable embedder (OpenAI, Voyage, Hugging Face, Ollama, or your own).
3. Runs a nearest-neighbor cosine similarity search in Postgres via pgvector.
4. Returns a cached response if a match passes the similarity threshold, or calls your route handler and stores the new response.

**Python is the glue, not the bottleneck.** Every expensive operation is offloaded:

| What | Where it runs |
|---|---|
| Cosine / ANN vector similarity | Postgres + pgvector (C, indexed) |
| Embedding generation | Your provider's API (I/O, not CPU) |
| Response blob storage and retrieval | Postgres rows or Redis (C clients) |
| HTTP proxying | `aiohttp.ClientSession` (async I/O, optional `proxy` extra) |

Because all meaningful work is either I/O-bound (GIL released) or executing inside a C extension, Python is never the ceiling even under high concurrency with a single `uvicorn` worker.

---

## Install

```bash
pip install fastapi-semcache
```

**`SEMANTIC_CACHE_PG_URI`** (PostgreSQL connection string with pgvector) is the only required environment variable. Everything else has a sensible default.

### Optional extras

| Extra | Installs | Use when |
|---|---|---|
| `proxy` | `fastapi`, `aiohttp` | `create_semantic_cache_proxy_app` |
| `embed-openai` | `openai`, `tiktoken` | `embedder_type="openai"` |
| `embed-voyage` | `voyageai`, `aiohttp` | `embedder_type="voyage"` |
| `embed-cohere` | `cohere` | `embedder_type="cohere"` |
| `embed-huggingface` | `sentence-transformers`, `torch` | `embedder_type="huggingface"` |
| `embed-ollama` | `openai` | `embedder_type="ollama"` |
| `redis` | `redis` | `SEMANTIC_CACHE_REDIS_URI` is set |

The core wheel installs **Starlette** and **psycopg** only. Declare **`fastapi`** in your own project for **`FastAPI()`** apps; the middleware is Starlette/ASGI middleware. **`fastapi`** is pulled in by the **`proxy`** extra for reverse proxy mode.

Extras can be combined:

```bash
pip install "fastapi-semcache[redis,embed-openai]"
```

For GPU (CUDA) PyTorch with the Hugging Face extra, pass PyTorch's wheel index:

```bash
pip install "fastapi-semcache[embed-huggingface]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

---

## Quickstart

### FastAPI middleware

```python
from typing import Any

from fastapi import FastAPI

from semanticcache import SemanticCache, SemanticCacheMiddleware
from semanticcache.embedders import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    ...

app = FastAPI()
cache = SemanticCache(embedder=MyEmbedder(...))
app.add_middleware(SemanticCacheMiddleware, cache=cache)


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}
```

```bash
uvicorn mymodule:app --host 0.0.0.0 --port 8000
```

By default only `POST` requests are intercepted. Successful responses whose body parses as a JSON object are stored. Cache hits replay the original HTTP status and response headers.

### Reverse proxy

Use `create_semantic_cache_proxy_app` when you want a standalone hop in front of another service rather than importing routes into your FastAPI app. Install the **`proxy`** extra first (installs **`fastapi`** and **`aiohttp`**):

```bash
pip install "fastapi-semcache[proxy]"
```

```python
from semanticcache import SemanticCache, create_semantic_cache_proxy_app

cache = SemanticCache(embedder=MyEmbedder(...))  # or set SEMANTIC_CACHE_EMBEDDER_TYPE
app = create_semantic_cache_proxy_app(
    upstream="http://127.0.0.1:11434",
    cache=cache,
)
```

```bash
uvicorn mymodule:app --host 0.0.0.0 --port 8080
```

---

## Key concepts

### Similarity thresholds

`SemanticCache` uses a two-stage retrieval pipeline:

- **Stage 1** (`SEMANTIC_CACHE_THRESHOLD`, `SEMANTIC_CACHE_TOP_K_CANDIDATES`): fetches the top-k nearest neighbors from pgvector that meet the primary similarity gate.
- **Stage 2** (`SEMANTIC_CACHE_REJECTION_THRESHOLD`): optionally applies a stricter cutoff on those candidates before serving a hit.

See [Cache Tuning](cache-tuning.md) for concrete configuration examples.

### Embedders

The default mode is **custom**: subclass `BaseEmbedder` and pass your instance to `SemanticCache(embedder=...)`. To use a built-in backend instead, set `SEMANTIC_CACHE_EMBEDDER_TYPE` to `openai`, `cohere`, `voyage`, `huggingface`, or `ollama`; `get_embedder()` then constructs the matching embedder when you omit `embedder=`:

```python
cache = SemanticCache(embedder=MyEmbedder(...), settings=get_cache_settings())
```

See [Embedders](embedders.md) for the full contract and built-in options.

### Cache scope and tenant isolation

By default (`SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=false`), the cache uses one shared bucket (single-tenant). For multi-tenant isolation, set `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE=true` and supply a server-side `extract_scope` that derives scope from authenticated identity. Do not rely on client-controlled `X-Semantic-Cache-Scope` or JSON `cache_scope` / `tenant_id` alone; clients can forge those values.

Middleware warning logs also avoid prompt text. Cache read failures log `route`,
`scope`, `request_id`, and a keyed digest of the composed lookup text. Set
`SEMANTIC_CACHE_LOG_DIGEST_KEY` when you want those digests to stay stable across
process restarts.

```python
from semanticcache.middleware.core.extractors import trusted_extract_scope_from_server_side

async def extract_scope(request, body: bytes) -> str | None:
    return await trusted_extract_scope_from_server_side(request)

app.add_middleware(SemanticCacheMiddleware, cache=cache, extract_scope=extract_scope)
app.add_middleware(YourAuthMiddleware)
```

### Storage

- **Postgres + pgvector**: always required. Each embedder configuration gets its own table (scoped by model id and vector dimension) created automatically on first use.
- **Redis** (optional): TTL-backed response blob cache. Install the `redis` extra and set `SEMANTIC_CACHE_REDIS_URI`. If unset, responses are stored in Postgres only.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_PG_URI` | _(required)_ | PostgreSQL connection string |
| `SEMANTIC_CACHE_PG_ENSURE_SCHEMA` | `true` | Create cache tables and indexes on first use; set `false` for externally managed schema |
| `SEMANTIC_CACHE_EMBEDDER_TYPE` | `custom` | `custom` (default): pass `embedder=` to `SemanticCache`. Built-in: `openai`, `cohere`, `voyage`, `huggingface`, `ollama`. `huggingface` loads PyTorch in-process; use hosted backends in production. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | Primary cosine similarity gate \[0.0, 1.0] |
| `SEMANTIC_CACHE_TOP_K_CANDIDATES` | `1` | Max nearest-neighbor candidates from pgvector |
| `SEMANTIC_CACHE_REJECTION_THRESHOLD` | _(unset)_ | Optional stricter second-stage cutoff |
| `SEMANTIC_CACHE_PGVECTOR_HNSW_M` | `16` | HNSW graph connectivity for new pgvector indexes; existing indexes keep the old value until rebuilt |
| `SEMANTIC_CACHE_PGVECTOR_HNSW_EF_CONSTRUCTION` | `64` | HNSW build candidate list size for new pgvector indexes; existing indexes keep the old value until rebuilt |
| `SEMANTIC_CACHE_PGVECTOR_HNSW_EF_SEARCH` | _(unset)_ | Optional default query-time HNSW search breadth; safe to change without rebuilding the index |
| `SEMANTIC_CACHE_REDIS_URI` | _(empty)_ | Redis URI; omit for Postgres-only mode |
| `SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE` | `false` | Require a non-empty scope on every request (multi-tenant) |
| `SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS` | `false` | Cache requests that include an `Authorization` header |
| `SEMANTIC_CACHE_LOG_DIGEST_KEY` | _(per-process random)_ | Secret used to derive HMAC digests for prompt-derived log fields; set explicitly for stable correlation across restarts |
| `SEMANTIC_CACHE_RESPONSE_MODE` | `buffered` | Miss delivery mode (`buffered` or `tee`) |
| `SEMANTIC_CACHE_HIT_RESPONSE_MODE` | _(auto)_ | Hit delivery mode (`single` or `stream`) |
| `SEMANTIC_CACHE_PG_TTL_DAYS` | _(unset)_ | Fractional days before Postgres rows expire |
| `SEMANTIC_CACHE_EMBED_TIMEOUT_SECONDS` | _(unset)_ | Fail-fast budget for embedder calls |
| `SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS` | _(unset)_ | Fail-fast budget for Postgres / Redis operations |
| `SEMANTIC_CACHE_UPSTREAM_TIMEOUT_SECONDS` | _(unset)_ | Fail-fast budget for upstream ASGI calls |
| `SEMANTIC_CACHE_MAX_BODY_BYTES` | `10485760` | Request and response body size cap (10 MiB) |

---

## Package names

The PyPI distribution and GitHub repository are **`fastapi-semcache`**. The import package is **`semanticcache`** (`fastapi_semcache` is available as an alias).

---

## Requirements

Python 3.12+. Postgres with the [pgvector](https://github.com/pgvector/pgvector) extension.

## License

[Apache-2.0](https://github.com/axm1647/fastapi-semcache/blob/main/LICENSE).
