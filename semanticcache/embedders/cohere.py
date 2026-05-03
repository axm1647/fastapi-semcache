from typing import override

from ._base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """
    This class will use openai library. It exists for user distinction but will be
    fundamentally the same as the OpenAIEmbedder
    """

    raise NotImplementedError()

    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()
