import torch

from cjw.utilities.embedding.Embedding import Embedding


class BertEmbedding(Embedding):
    async def embed(self, text: str) -> torch.Array:
        ...
