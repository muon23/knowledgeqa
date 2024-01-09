from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Embedding(ABC):

    @abstractmethod
    async def embed(self, text: str) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    async def embed1(self, text: str) -> Optional[np.ndarray]:
        pass
