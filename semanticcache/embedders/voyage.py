from typing import override

from ._base import BaseEmbedder


class VoyageEmbedder(BaseEmbedder):
    raise NotImplementedError()

    @override
    async def embed(self, texts: list[str]) -> list[list[str]]:
        raise NotImplementedError()
