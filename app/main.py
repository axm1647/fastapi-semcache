"""ASGI entrypoint for the semantic cache reverse proxy service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from semanticcache import (
    SemanticCache,
    create_semantic_cache_proxy_app,
    get_cache_settings,
)


class ProxyAppSettings(BaseSettings):
    """Load proxy deployment settings from the environment."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="SEMANTIC_CACHE_PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    upstream: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL of the backend API to forward requests to.",
    )


def _build_app() -> FastAPI:
    """Create the ASGI app and chain semantic cache shutdown after the proxy lifespan.

    Returns:
        FastAPI application passed to uvicorn.
    """
    settings = ProxyAppSettings()
    cache = SemanticCache(settings=get_cache_settings())
    app = create_semantic_cache_proxy_app(
        upstream=settings.upstream,
        cache=cache,
    )
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
    return app


app = _build_app()
