from typing import TYPE_CHECKING

from semanticcache import SemanticCache, create_semantic_cache_proxy_app

if TYPE_CHECKING:
    from fastapi import FastAPI


cache = SemanticCache()
upstream = "https://api.openai.com/v1"

app: "FastAPI" = create_semantic_cache_proxy_app(upstream=upstream, cache=cache)
