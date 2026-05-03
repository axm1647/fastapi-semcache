from .config import get_cache_settings


class SemanticCache:
    """
    Semantic caching engine:
        - TODO: write docstring
    """

    threshold: float
    pg_uri: str
    redis_uri: str

    def __init__(
        self,
        threshold: float | None = None,
        pg_uri: str | None = None,
        redis_uri: str | None = None,
    ) -> None:
        settings = get_cache_settings()

        self.threshold = threshold if threshold is not None else settings.threshold
        self.pg_uri = pg_uri if pg_uri is not None else settings.pg_uri
        self.redis_uri = redis_uri if redis_uri is not None else settings.redis_uri

    async def get(self):
        raise NotImplementedError()

    async def put(self):
        raise NotImplementedError()

    async def close(self):
        raise NotImplementedError()
