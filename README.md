# semanticcache

Semantic caching middleware and reverse proxy for APIs and LLMs, with local embeddings, pgvector similarity search, and Redis-backed response caching.

The PyPI distribution name is **`semanticcache-py`** (the import package remains `semanticcache`).

## Why SemanticCache?

SemanticCache is designed for direct integration into modern Python API stacks with no refactoring needed. It keeps the caching path simple and gives you explicit control over embeddings, vector search, and cache behavior.

It includes **FastAPI** middleware as a first-class integration path and can also run as a reverse proxy in front of an upstream API or LLM service. **Django** and **Flask** middleware are planned for a future release so you can hook semantic caching into those stacks the same way as FastAPI.

## What is implemented

- **Local embeddings** via Sentence Transformers (`embedder_type="local"`).
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

## Future support

- **Django** and **Flask** middleware for in-app semantic caching (not yet shipped. same role as the FastAPI middleware).

Embeddings from the following providers are planned:

- Cohere
- Voyage

## Quick start

```python
from semanticcache import SemanticCache, create_semantic_cache_proxy_app

cache = SemanticCache()
app = create_semantic_cache_proxy_app(
    upstream="http://127.0.0.1:11434",
    cache=cache,
)
```

Run with:

```bash
uvicorn mymodule:app --host 0.0.0.0 --port 8080
```

## Reverse proxy

Point clients at the proxy and configure Postgres, Redis, and the upstream base URL.

This repository includes a small ASGI app at `app/main.py` (import `app` for uvicorn). Set **`SEMANTIC_CACHE_PROXY_UPSTREAM`** to the backend base URL; the default is `http://127.0.0.1:11434`.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

See `create_semantic_cache_proxy_app` in `semanticcache.proxy` for timeout, TLS verification, `httpx_client_kwargs`, and middleware options such as `path_prefix` and `extract_query`.

## Install

```bash
pip install semanticcache-py
```

Optional extras:

- `embed-local` / `embed-local-cpu`: Sentence Transformers with **CPU** PyTorch.
- `embed-local-gpu`: Sentence Transformers with a CUDA-enabled PyTorch install.
- `embed-openai`: OpenAI embeddings (`openai`, `tiktoken`).

### CPU

```bash
pip install "semanticcache-py[embed-local-cpu]"
# or: pip install "semanticcache-py[embed-local]"
```

### GPU

Pick a CUDA version that matches your system from [PyTorch Get Started](https://pytorch.org/get-started/locally/), then install with that index so pip selects CUDA wheels.

```bash
pip install "semanticcache-py[embed-local-gpu]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

### OpenAI embeddings

Install the OpenAI extra so `embedder_type="openai"` works (pulls `openai` and `tiktoken`). Set `OPENAI_API_KEY` in your environment.

```bash
pip install "semanticcache-py[embed-openai]"
```

## Requirements

Python 3.12+.

## Links

- Repository: [SemanticCache-py](https://github.com/axm1647/SemanticCache-py)

## License

Apache-2.0. See `LICENSE`.
