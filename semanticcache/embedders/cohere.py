from typing import override

from .openai import OpenAIEmbedder


class CohereEmbedder(OpenAIEmbedder):
    """
    This class will use openai library. It exists for user distinction but will be
    fundamentally the same as the OpenAIEmbedder
    """
    @override
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()
