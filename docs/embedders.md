# Custom embedders and minimal installs

The pypi package **`semanticcache-py`** installs core runtime dependencies only (FastAPI, HTTPX, Postgres, Redis, settings). Optional extras such as `embed-openai` and `embed-huggingface` pull in vendor-specific stacks.

If you want to avoid those stacks, or you already host embeddings elsewhere, implement a small class against **`BaseEmbedder`** and pass it into **`SemanticCache(embedder=...)`**. No embedding extra is required for that path.

## Install (core only)

```bash
pip install semanticcache-py
```

You still need Postgres with **pgvector** and, if you use it, Redis. You do **not** need `embed-openai`, `embed-huggingface`, or similar extras for a custom embedder.

## Contract: `BaseEmbedder`

Import the abstract base from the embedders package:

```python
from semanticcache.embedders import BaseEmbedder
```

Your subclass must provide:

| Member | Role |
| --- | --- |
| **`embedding_dim`** (property) | Length of each dense vector returned by `embed`. Must stay fixed for a given `cache_namespace` so the pgvector table dimension matches. |
| **`cache_namespace`** (property) | Stable string that identifies this embedding setup (model id, version, dimension, anything that should not share storage with a different setup). Used to derive the pgvector table name and Redis key prefix. |
| **`embed(texts)`** (async) | Given `list[str]`, return `list[list[float]]`: one vector per string, **same order and length** as `texts`. For an empty input list, return an empty list. |

Batching, retries, and timeouts are your responsibility inside `embed`.

## Wiring `SemanticCache`

Pass your instance as the keyword-only argument **`embedder`**. The factory **`get_embedder(settings)`** is not used when `embedder` is set, so **`SEMANTIC_CACHE_EMBEDDER_TYPE`** does not select your implementation.

```python
from semanticcache import SemanticCache, get_cache_settings

cache = SemanticCache(embedder=MyEmbedder(...), settings=get_cache_settings())
```

Optional: pass **`embedding_dim=`** to assert it matches `embedder.embedding_dim` (a mismatch raises **`ValueError`**).

### `CacheResult.source` and settings

`CacheResult.source` is still derived from **`CacheSettings.embedder_type`** (environment **`SEMANTIC_CACHE_EMBEDDER_TYPE`**) for hits and misses, not from your custom class. If you rely on that field for metrics, either align `embedder_type` with how you want logs labeled or treat `source` as configuration metadata only when using a custom embedder.

## Example: HTTP example API with HTTPX

**HTTPX** is already a core dependency. The snippet below assumes your service accepts `POST /embed` with body `{"texts": ["...", ...]}` and returns JSON like `{"vectors": [[float, ...], ...]}`. Rename paths and keys to match your API.

```python
from typing import override

import httpx
from semanticcache.embedders import BaseEmbedder


class HttpExampleeEmbedder(BaseEmbedder):
    """Minimal example: delegate embeddings to your own HTTP service."""

    def __init__(
        self,
        *,
        base_url: str,
        embedding_dim: int,
        cache_namespace: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._embedding_dim = embedding_dim
        self._cache_namespace = cache_namespace

    @property
    @override
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    @override
    def cache_namespace(self) -> str:
        return self._cache_namespace

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(base_url=self._base_url, timeout=60.0) as client:
            response = await client.post("/embed", json={"texts": texts})
            response.raise_for_status()
            payload = response.json()
        vectors: list[list[float]] = payload["vectors"]
        if len(vectors) != len(texts):
            msg = "embedding API returned wrong number of vectors"
            raise RuntimeError(msg)
        for row in vectors:
            if len(row) != self._embedding_dim:
                msg = "embedding vector length does not match embedding_dim"
                raise RuntimeError(msg)
        return vectors
```

Usage:

```python
from semanticcache import SemanticCache, get_cache_settings

embedder = HttpExampleEmbedder(
    base_url="http://127.0.0.1:9000",
    embedding_dim=768,
    cache_namespace="my-team-embed-v1-d768",
)
cache = SemanticCache(embedder=embedder, settings=get_cache_settings())
```

Pick **`cache_namespace`** so it changes whenever model, pooling, or vector width changes; otherwise you risk reading incompatible rows from an old table.
> Pick cache_namespace so it changes whenever model, pooling, or vector width changes; otherwise you risk reading incompatible rows from an old table. If storage is shared, include an application or environment prefix so your namespace cannot match another service’s built-in vendor:model:dimensions string by accident.

## Reusing a long-lived HTTP client

Opening a client per `embed` call is simple but not ideal under load. You can hold an **`httpx.AsyncClient`** on the embedder and close it when your app shuts down (for example in a FastAPI lifespan handler). **`SemanticCache.close`** does not close your embedder.

## See also

Built-in embedders and optional extras are described in the repository **`README.md`** (Install section).
