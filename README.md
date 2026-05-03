# semanticcache

Semantic caching for APIs and LLMs: local embeddings, vector storage (pgvector), response caching (Redis), and optional FastAPI middleware.

The PyPI distribution name is **`semanticcache-py`** (the import package remains `semanticcache`).

## Why SemanticCache?

SemanticCache avoids LangChain and other orchestration frameworks so integration stays simple and you keep direct control over embeddings, vector search, and cache behavior. It ships **FastAPI** middleware as a first-class integration path; `fastapi` is a core dependency of this package.

It fits performance-sensitive API and LLM workloads where low latency and predictable behavior matter.

## What is implemented

- **Local embeddings** (Sentence Transformers): supported via `embedder_type='local'` and the `embed-local*` extras.
- **PostgreSQL + pgvector** and **Redis** response caching: supported (see `SemanticCache` and settings).
- **Reverse proxy**: `create_semantic_cache_proxy_app()` exposes a FastAPI app that forwards HTTP requests to a configurable upstream URL with the same semantic caching behavior as `SemanticCacheMiddleware` on your own app.
- **OpenAI embeddings**: **not implemented yet.** Choosing `embedder_type='openai'` will raise until an OpenAI embedder is added. The `embed-openai` extra only installs optional dependencies (`openai`, `tiktoken`) for when that support exists or for your own wiring.

### Reverse proxy

Point clients at the proxy; configure Postgres, Redis, and the backend base URL. Example:

```python
from semanticcache import SemanticCache, create_semantic_cache_proxy_app

cache = SemanticCache()
app = create_semantic_cache_proxy_app(
    upstream="http://127.0.0.1:11434",
    cache=cache,
)

# uvicorn mymodule:app --host 0.0.0.0 --port 8080
```

This repository includes a small ASGI app at `app/main.py` (import `app` for uvicorn). Set **`SEMANTIC_CACHE_PROXY_UPSTREAM`** to the backend base URL (default `http://127.0.0.1:11434`). Example:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

See `create_semantic_cache_proxy_app` in `semanticcache.proxy` for timeout, TLS verification, `httpx_client_kwargs` (for example a mock `transport` in tests), and middleware options (`path_prefix`, `extract_query`, etc.).

## Install

```bash
pip install semanticcache-py
```

Optional extras:

- `embed-local` / `embed-local-cpu`: local embeddings via Sentence Transformers with **CPU** PyTorch (default PyPI wheels)
- `embed-local-gpu`: same libraries; install with PyTorch’s **CUDA** wheel index so `torch` resolves to a GPU build
- `embed-openai`: installs `openai` and `tiktoken` only; built-in OpenAI embedding support is not implemented yet (see above)

### CPU (Sentence Transformers + CPU PyTorch)

```bash
pip install "semanticcache-py[embed-local-cpu]"
# or: pip install "semanticcache-py[embed-local]"
```

### GPU (Sentence Transformers + CUDA PyTorch)

Pick a CUDA version that matches your driver and OS from [PyTorch Get Started](https://pytorch.org/get-started/locally/), then install with that index so pip selects CUDA wheels (example for CUDA 12.4):

```bash
pip install "semanticcache-py[embed-local-gpu]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Use the URL shown on the PyTorch site for your platform; the extra name is the same (`embed-local-gpu`).

## Requirements

Python 3.12+.

## Links

- Repository: [SemanticCache-py](https://github.com/axm1647/SemanticCache-py)

## License

Apache-2.0. See `LICENSE`.
