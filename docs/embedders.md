# Custom embedders and minimal installs

The PyPI package **`fastapi-semcache`** installs core runtime dependencies only (Starlette and Postgres via `psycopg`). Optional extras such as `proxy`, `embed-openai`, `embed-cohere`, `embed-voyage`, `embed-ollama`, `embed-huggingface` and `redis` pull in vendor-specific stacks. FastAPI apps should already declare **`fastapi`** in their own project; the **`proxy`** extra installs **`fastapi`** for **`create_semantic_cache_proxy_app`**.

If you want to avoid those stacks, or you already host embeddings elsewhere, implement a small class against **`BaseEmbedder`** and pass it into **`SemanticCache(embedder=...)`**. No embedding extra is required for that path.

## Built-in embedders: `get_embedder` vs constructor arguments

**`CacheSettings.embedder_type`** defaults to **`custom`**. In that mode you **must** pass **`embedder=`** to **`SemanticCache`** with a **`BaseEmbedder`** subclass; **`get_embedder()`** is not used.

**`get_embedder(settings)`** runs only when you omit **`embedder=`** on **`SemanticCache`** and **`embedder_type`** is a built-in backend (set **`SEMANTIC_CACHE_EMBEDDER_TYPE`** to **`openai`**, **`cohere`**, **`voyage`**, **`huggingface`**, or **`ollama`**). It reads the matching settings fields. It constructs **`SBERTEmbedder`** or **`OpenAIEmbedder`** with **no** **`model_name`**, **`dimensions`**, **`base_url`**, or other constructor overrides for those two backends (they use **class defaults**, for example **`text-embedding-3-small`** / **`1536`** on **`OpenAIEmbedder`**, or **`sentence-transformers/all-MiniLM-L6-v2`** on **`SBERTEmbedder`**).

When **`embedder_type`** is **`ollama`**, settings **must** include **`ollama_embedding_model`** and **`ollama_embedding_dimensions`** (environment **`SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL`** and **`SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS`**). There is no safe library default across Qwen, Nomic, and other models; the declared width must match the running model and your pgvector column.

There is **no** separate environment variable today for “which embedding model id” on the stock factory path for **`OpenAIEmbedder`** / **`SBERTEmbedder`**. Changing the embedding model or dimensions for those backends is normal configuration: import **`OpenAIEmbedder`** or **`SBERTEmbedder`**, pass the constructor arguments you need, and wire **`SemanticCache(embedder=..., settings=...)`** as below.

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

Pass any **`BaseEmbedder`** instance (built-in **`OpenAIEmbedder`**, **`SBERTEmbedder`**, **`OllamaEmbedder`**, or your own subclass) as the keyword-only argument **`embedder`**. The factory **`get_embedder(settings)`** is not used when **`embedder`** is set, so **`SEMANTIC_CACHE_EMBEDDER_TYPE`** does not choose that instance.

```python
from semanticcache import SemanticCache, get_cache_settings

cache = SemanticCache(embedder=MyEmbedder(...), settings=get_cache_settings())
```

Optional: pass **`embedding_dim=`** to assert it matches `embedder.embedding_dim` (a mismatch raises **`ValueError`**).

When you use the built-in Hugging Face backend through **`get_embedder(settings)`**, **`SBERTEmbedder`** receives the token from **`settings.hugging_face_api_key`**. **`SBERTEmbedder`** does not re-read global settings on its own.

**Production note:** **`SBERTEmbedder`** loads **sentence-transformers** and **PyTorch** inside your app process. That adds significant memory and CPU/GPU overhead compared with hosted APIs. It is intended for local development and tests. On first construction, the library emits a one-time **`UserWarning`** recommending **`openai`**, **`cohere`**, **`voyage`**, **`ollama`**, or a custom **`BaseEmbedder`** for deployed workloads.

### `CacheResult.source` and settings

**`CacheResult.source`** is still derived from **`CacheSettings.embedder_type`** (environment **`SEMANTIC_CACHE_EMBEDDER_TYPE`**) for hits and misses, not from the concrete embedder class or **`model_name`**. If you rely on that field for metrics, either align **`embedder_type`** with the backend you instantiated (**`huggingface`** vs **`openai`**) or treat **`source`** as configuration metadata only when **`embedder`** was passed explicitly.

## Example: HTTP embedding API

Delegate embeddings to your own HTTP service. Both snippets below assume `POST /embed` with body `{"texts": ["...", ...]}` and JSON `{"vectors": [[float, ...], ...]}`. Rename paths and keys to match your API.

### HTTPX

Install **`httpx`** in your app (or add it to your project dependencies).

```python
from typing import override

import httpx
from semanticcache.embedders import BaseEmbedder


class HttpExampleHttpxEmbedder(BaseEmbedder):
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

### aiohttp

Install **`aiohttp`** in your app, or use an extra that already includes it (for example **`fastapi-semcache[proxy]`** or **`embed-voyage`**).

```python
from typing import Any, override

import aiohttp
from semanticcache.embedders import BaseEmbedder


class HttpExampleAiohttpEmbedder(BaseEmbedder):
    """Minimal example: delegate embeddings to your own HTTP service (aiohttp)."""

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
        timeout = aiohttp.ClientTimeout(total=60.0)
        async with aiohttp.ClientSession(
            base_url=self._base_url,
            timeout=timeout,
        ) as session:
            async with session.post("/embed", json={"texts": texts}) as response:
                response.raise_for_status()
                payload: dict[str, Any] = await response.json()
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

### Usage

```python
from semanticcache import SemanticCache, get_cache_settings

embedder = HttpExampleAiohttpEmbedder(
    base_url="http://127.0.0.1:9000",
    embedding_dim=768,
    cache_namespace="my-team-embed-v1-d768",
)
# Or: HttpExampleHttpxEmbedder(...)
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

## VoyageEmbedder (`embed-voyage`)

**`VoyageEmbedder`** (optional extra **`embed-voyage`**) sends embedding requests asynchronously to `https://api.voyageai.com/v1/embeddings` using **`aiohttp`**, as recommended by Voyage for async workloads. Before each batch, it validates inputs locally via **`voyageai.Client.tokenize`** (which uses Voyage's Hugging Face tokenizer - no network call).

Install with:

```bash
pip install 'fastapi-semcache[embed-voyage]'
```

### Constructor

```python
VoyageEmbedder(
    model_name="voyage-4",
    *,
    dimensions=1024,
    output_dimension=None,
    input_type=None,
    api_key=None,
)
```

- **`model_name`**: Voyage model id. Recommended: `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-3`, `voyage-3.5`, `voyage-code-3`. Defaults to `voyage-4`.
- **`dimensions`**: Storage and validation width. Must match the model's actual output (or `output_dimension` when set).
- **`output_dimension`**: When set, passed as `output_dimension` in the API request. Only supported by `voyage-4-*`, `voyage-3-large`, `voyage-3.5*`, and `voyage-code-3` (valid values: 256, 512, 1024, 2048). When used, set `dimensions` to the same value.
- **`input_type`**: Optional hint: `None`, `"query"`, or `"document"`. Use `"document"` when indexing content, `"query"` for lookup to improve retrieval accuracy. Passed directly to the API.
- **`api_key`**: Voyage API key. Defaults to the `VOYAGE_API_KEY` environment variable when omitted.

### Example

```python
from semanticcache import SemanticCache, get_cache_settings
from semanticcache.embedders import VoyageEmbedder

cache = SemanticCache(
    embedder=VoyageEmbedder(
        model_name="voyage-4-large",
        dimensions=1024,
        input_type="document",
        api_key=get_cache_settings().voyage_api_key,
    ),
    settings=get_cache_settings(),
)
```

### Through `get_embedder`

Set `SEMANTIC_CACHE_EMBEDDER_TYPE=voyage`. The following environment variables configure the factory path:

| Variable | Default | Description |
| --- | --- | --- |
| `VOYAGE_API_KEY` / `SEMANTIC_CACHE_VOYAGE_API_KEY` | `None` | API key |
| `SEMANTIC_CACHE_VOYAGE_EMBEDDING_MODEL` | `voyage-4` | Model id |
| `SEMANTIC_CACHE_VOYAGE_EMBEDDING_DIMENSIONS` | `1024` | Vector width |
| `SEMANTIC_CACHE_VOYAGE_INPUT_TYPE` | `None` | `query`, `document`, or unset |

### Notes

- The `aiohttp.ClientSession` is created lazily on the first `embed()` call and shared across all requests. Call `await embedder.aclose()` on shutdown, or `await cache.close()` which invokes `aclose()` when the embedder implements it.
- Token validation via `voyageai.Client.tokenize` is a local CPU operation - it loads the model's Hugging Face tokenizer on first call.
- Batches are capped at 1,000 texts per request (Voyage's documented hard limit).

## CohereEmbedder (`embed-cohere`)

**`CohereEmbedder`** (optional extra **`embed-cohere`**) calls Cohere's embed API through the official **`cohere.AsyncClient`**. By default it uses **`AsyncClient.embed`**, which batches requests at 96 texts per call. When **`output_dimension`** is set (supported on **`embed-v4`** and newer), it uses **`AsyncClient.v2.embed`** with manual batching so the reduced width is sent to the API.

Install with:

```bash
pip install 'fastapi-semcache[embed-cohere]'
```

### Constructor

```python
CohereEmbedder(
    model_name="embed-v4.0",
    *,
    dimensions=1536,
    input_type="search_document",
    output_dimension=None,
    truncate=None,
    api_key=None,
    base_url=None,
)
```

- **`model_name`**: Cohere embed model id (for example **`embed-v4.0`**, **`embed-english-v3.0`**, **`embed-multilingual-v3.0`**). Defaults to **`embed-v4.0`**.
- **`dimensions`**: Storage and validation width. Must match the model's actual output (1536 for **`embed-v4.0`** by default, or **`output_dimension`** when set).
- **`input_type`**: Required hint for embed v3+: **`search_document`**, **`search_query`**, **`classification`**, or **`clustering`**. Use **`search_document`** when indexing content and **`search_query`** for lookup. Defaults to **`search_document`**.
- **`output_dimension`**: When set, passed to the v2 API. Only supported by **`embed-v4`** and newer (256, 512, 1024, 1536). Set **`dimensions`** to the same value.
- **`truncate`**: Optional **`NONE`**, **`START`**, or **`END`** for over-length inputs. When omitted, the API default applies.
- **`api_key`**: Cohere API key. Defaults to **`COHERE_API_KEY`** when omitted.
- **`base_url`**: Optional API base URL for enterprise or proxy deployments.

### Example

```python
from semanticcache import SemanticCache, get_cache_settings
from semanticcache.embedders import CohereEmbedder

cache = SemanticCache(
    embedder=CohereEmbedder(
        model_name="embed-v4.0",
        dimensions=1536,
        input_type="search_document",
        api_key=get_cache_settings().cohere_api_key,
    ),
    settings=get_cache_settings(),
)
```

### Through `get_embedder`

Set **`SEMANTIC_CACHE_EMBEDDER_TYPE=cohere`**. The following environment variables configure the factory path:

| Variable | Default | Description |
| --- | --- | --- |
| `COHERE_API_KEY` / `SEMANTIC_CACHE_COHERE_API_KEY` | `None` | API key |
| `SEMANTIC_CACHE_COHERE_EMBEDDING_MODEL` | `embed-v4.0` | Model id |
| `SEMANTIC_CACHE_COHERE_EMBEDDING_DIMENSIONS` | `1536` | Vector width |
| `SEMANTIC_CACHE_COHERE_INPUT_TYPE` | `search_document` | `search_document`, `search_query`, etc. |

### Notes

- Call **`await embedder.aclose()`** on shutdown, or **`await cache.close()`**, which invokes **`aclose()`** when the embedder implements it.
- Without **`output_dimension`**, the SDK's built-in batching on **`AsyncClient.embed`** applies (96 texts per request).

## OllamaEmbedder (`embed-ollama`)

**`OllamaEmbedder`** uses the official **`openai`** **`AsyncOpenAI`** client against Ollama’s **OpenAI-compatible** embeddings endpoint (configure **`base_url`** with the **`/v1`** suffix, for example **`http://127.0.0.1:11434/v1`**). Install **`fastapi-semcache[embed-ollama]`**.

You must pass **`model_name`** and **`dimensions`** explicitly. After each response, the backend checks that every vector length equals **`dimensions`** so misconfiguration fails fast.

Optional **`api_key`**: when omitted, a placeholder Bearer value is sent so the SDK does not pick up **`OPENAI_API_KEY`** from the environment; set **`api_key`** when your Ollama deployment requires auth.

Through **`get_embedder(settings)`**, **`SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL`**, **`SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS`**, **`SEMANTIC_CACHE_OLLAMA_BASE_URL`**, and **`OLLAMA_API_KEY`** / **`SEMANTIC_CACHE_OLLAMA_API_KEY`** populate those fields.

## Reusing a long-lived HTTP client

Opening a client per `embed` call is simple but not ideal under load. You can hold an **`httpx.AsyncClient`** or **`aiohttp.ClientSession`** on the embedder and close it when your app shuts down (for example in a FastAPI lifespan handler). Implement **`aclose()`** on custom embedders with long-lived clients; **`SemanticCache.close()`** awaits it when present.

## See also

Built-in embedders and optional extras are described in the repository **`README.md`** (Install section).
