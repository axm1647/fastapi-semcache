"""Minimal standalone reverse proxy: OpenAI-compatible API with semantic caching.

Run from the repo root with Postgres (pgvector) and optional Redis configured via
``SEMANTIC_CACHE_*`` env vars. Clients send ``Authorization: Bearer ...``; the proxy
forwards headers to the upstream so your OpenAI key reaches ``api.openai.com``.
Because the middleware bypasses cache for authorized requests by default, set
``SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS=true`` (or pass
``CacheSettings(cache_authorized_requests=True)``) when running this proxy if you
want requests with ``Authorization`` to produce cache hits.

Example:

    uv run python examples/proxy.py

Or import ``app`` and serve with any ASGI server.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from semanticcache import (
    SemanticCache,
    create_semantic_cache_proxy_app,
    get_cache_settings,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

# Proxy traffic usually includes Authorization headers. Opt in with
# SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS=true if you want those requests cached.
cache = SemanticCache(settings=get_cache_settings())
upstream = "https://api.openai.com/v1"

app: FastAPI = create_semantic_cache_proxy_app(upstream=upstream, cache=cache)
inner_lifespan = app.router.lifespan_context


@asynccontextmanager
async def chained_lifespan(application: FastAPI):
    """Run httpx client startup/shutdown, then close pg/redis on shutdown.

    Args:
        application: ASGI application instance from the lifespan scope.

    Yields:
        Control after proxy startup until shutdown begins.
    """
    async with inner_lifespan(application):
        yield
    await cache.close()


app.router.lifespan_context = chained_lifespan


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
