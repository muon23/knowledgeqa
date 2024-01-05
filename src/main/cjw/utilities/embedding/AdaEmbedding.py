import torch

from cjw.utilities.embedding.Embedding import Embedding


class AdaEmbedding(Embedding):
    async def embed(self, text: str) -> torch.Array:
        ...
