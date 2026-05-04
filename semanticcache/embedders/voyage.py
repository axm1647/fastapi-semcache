from typing import override

from ._base import BaseEmbedder


class VoyageEmbedder(BaseEmbedder):

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()
