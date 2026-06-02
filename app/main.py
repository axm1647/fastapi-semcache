"""ASGI entrypoint for the semantic cache reverse proxy service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from semanticcache import (
    SemanticCache,
    create_semantic_cache_proxy_app,
    get_cache_settings,
)


def _build_app() -> FastAPI:
    """Create the ASGI app and chain semantic cache shutdown after the proxy lifespan.

    Returns:
        FastAPI application passed to uvicorn.
    """
    upstream = os.getenv("SEMANTIC_CACHE_PROXY_UPSTREAM", "http://127.0.0.1:11434")
    cache = SemanticCache(settings=get_cache_settings())
    app = create_semantic_cache_proxy_app(
        upstream=upstream,
        cache=cache,
    )
    inner_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def chained_lifespan(application: FastAPI):
        """Run aiohttp session startup/shutdown, then close pg/redis on shutdown.

        Args:
            application: ASGI application instance from the lifespan scope.

        Yields:
            Control after proxy startup until shutdown begins.
        """
        async with inner_lifespan(application):
            yield
        await cache.close()

    app.router.lifespan_context = chained_lifespan
    return app


app = _build_app()
