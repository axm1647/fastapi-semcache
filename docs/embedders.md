# Custom embedders and minimal installs

The PyPI package **`fastapi-semcache`** installs core runtime dependencies only (FastAPI, HTTPX, Postgres, Redis, settings). Optional extras such as `embed-openai` and `embed-huggingface` pull in vendor-specific stacks.

If you want to avoid those stacks, or you already host embeddings elsewhere, implement a small class against **`BaseEmbedder`** and pass it into **`SemanticCache(embedder=...)`**. No embedding extra is required for that path.

## Built-in embedders: `get_embedder` vs constructor arguments

**`get_embedder(settings)`** (used automatically when you omit **`embedder=`** on **`SemanticCache`**) only reads **`CacheSettings.embedder_type`** (environment **`SEMANTIC_CACHE_EMBEDDER_TYPE`**) and the matching API key field. It constructs **`SBERTEmbedder`** or **`OpenAIEmbedder`** with **no** **`model_name`**, **`dimensions`**, **`base_url`**, or other constructor overrides. Those embedders then use their **class defaults** (for example **`text-embedding-3-small`** / **`1536`** on **`OpenAIEmbedder`**, or **`sentence-transformers/all-MiniLM-L6-v2`** on **`SBERTEmbedder`**).

There is **no** separate environment variable today for “which embedding model id” on the stock factory path. Changing the embedding model or dimensions for a **built-in** embedder is normal configuration, not a custom embedder: import **`OpenAIEmbedder`** or **`SBERTEmbedder`**, pass the constructor arguments you need, and wire **`SemanticCache(embedder=..., settings=...)`** as below.

```python
from semanticcache import SemanticCache, get_cache_settings
from semanticcache.embedders import OpenAIEmbedder

cache = SemanticCache(
    embedder=OpenAIEmbedder(
        model_name="text-embedding-3-large",
        dimensions=3072,
        api_key=get_cache_settings().openai_api_key,
    ),
    settings=get_cache_settings(),
)
```

Use the same pattern for **`SBERTEmbedder(model_name="...", normalize_embeddings=..., api_key=...)`**. **`cache_namespace`** (and thus pgvector table routing) incorporates model id and dimensions, so a different **`model_name`** or width does not collide with another setup.

When you bypass **`get_embedder`**, **`SEMANTIC_CACHE_EMBEDDER_TYPE`** no longer selects the implementation; keep **`embedder_type`** aligned with reality if you rely on **`CacheResult.source`** (see **`CacheResult.source` and settings** below).

## Install (core only)

```bash
pip install fastapi-semcache
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

Pass any **`BaseEmbedder`** instance (built-in **`OpenAIEmbedder`** / **`SBERTEmbedder`** with non-default args, or your own subclass) as the keyword-only argument **`embedder`**. The factory **`get_embedder(settings)`** is not used when **`embedder`** is set, so **`SEMANTIC_CACHE_EMBEDDER_TYPE`** does not choose that instance.

```python
from semanticcache import SemanticCache, get_cache_settings

cache = SemanticCache(embedder=MyEmbedder(...), settings=get_cache_settings())
```

Optional: pass **`embedding_dim=`** to assert it matches `embedder.embedding_dim` (a mismatch raises **`ValueError`**).

When you use the built-in Hugging Face backend through **`get_embedder(settings)`**, **`SBERTEmbedder`** receives the token from **`settings.hugging_face_api_key`**. **`SBERTEmbedder`** does not re-read global settings on its own.

### `CacheResult.source` and settings

**`CacheResult.source`** is still derived from **`CacheSettings.embedder_type`** (environment **`SEMANTIC_CACHE_EMBEDDER_TYPE`**) for hits and misses, not from the concrete embedder class or **`model_name`**. If you rely on that field for metrics, either align **`embedder_type`** with the backend you instantiated (**`huggingface`** vs **`openai`**) or treat **`source`** as configuration metadata only when **`embedder`** was passed explicitly.

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

Pick **`cache_namespace`** so it changes whenever model, pooling, or vector width changes; otherwise you risk reading incompatible rows from an old table. If storage is shared, include an application or environment prefix so your namespace cannot match another service’s built-in **`vendor:model:dimensions`** string by accident.

## OpenAIEmbedder and `send_dimensions_to_api`

**`OpenAIEmbedder`** (optional extra **`embed-openai`**) maps the `dimensions` constructor argument to both **storage width** and, by default, the OpenAI **`embeddings.create`** request body. Some models (for example **`text-embedding-ada-002`**) return a **fixed** vector size and the API may **reject** a `dimensions` parameter. In that case, set **`send_dimensions_to_api=False`**. The library still uses **`dimensions`** for `embedding_dim`, validation, and `cache_namespace` - it only **omits** the field from the API call.

```python
from semanticcache.embedders import OpenAIEmbedder

# Fixed-size model: do not send "dimensions" to the API, but keep local width 1536.
ada = OpenAIEmbedder(
    model_name="text-embedding-ada-002",
    dimensions=1536,
    send_dimensions_to_api=False,
)
```

For **`text-embedding-3-small`** / **`text-embedding-3-large`**, the default **`send_dimensions_to_api=True`** is appropriate when you want a reduced **output** width (as supported by that model family).

## Reusing a long-lived HTTP client

Opening a client per `embed` call is simple but not ideal under load. You can hold an **`httpx.AsyncClient`** on the embedder and close it when your app shuts down (for example in a FastAPI lifespan handler). **`SemanticCache.close`** does not close your embedder.

## See also

Built-in embedders and optional extras are described in the repository **`README.md`** (Install section).
