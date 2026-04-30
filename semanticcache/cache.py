class SemanticCache:
    """
    Semantic caching engine:
        - TODO: write docstring
    """

    threshold: float
    vector_uri: str
    redis_uri: str

    def __init__(
        self,
        threshold: float = 0.9,
        vector_uri: str = "postgresql://user:pass@localhost:5433/semanticcache",
        redis_uri: str = "redis://localhost:6379/0",
    ) -> None:
        self.threshold = threshold
        self.vector_uri = vector_uri
        self.redis_uri = redis_uri

    async def get(self):
        raise NotImplementedError()

    async def put(self):
        raise NotImplementedError()

    async def close(self):
        raise NotImplementedError()
