from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from semanticcache import SemanticCache, create_semantic_cache_proxy_app

if TYPE_CHECKING:
    from fastapi import FastAPI


cache = SemanticCache()
upstream = "https://api.openai.com/v1"

app: "FastAPI" = create_semantic_cache_proxy_app(upstream=upstream, cache=cache)
inner_lifespan = app.router.lifespan_context


@asynccontextmanager
async def chained_lifespan(application: "FastAPI"):
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

    uvicorn.run(app)
