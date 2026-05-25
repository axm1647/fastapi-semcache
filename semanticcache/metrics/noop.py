class NoOpCollector:
    def record_cache_hit(self, *a, **kw) -> None:
        pass

    def record_cache_miss(self, *a, **kw) -> None:
        pass

    def record_put(self, *a, **kw) -> None:
        pass

    def record_timeout(self, *a, **kw) -> None:
        pass

    def record_error(self, *a, **kw) -> None:
        pass
