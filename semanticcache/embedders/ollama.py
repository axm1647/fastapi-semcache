from typing import override

from ._base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()
