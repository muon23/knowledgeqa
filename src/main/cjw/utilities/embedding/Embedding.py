from abc import ABC, abstractmethod
import torch


class Embedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def embed(self, text: str) -> torch.Array:
        pass
