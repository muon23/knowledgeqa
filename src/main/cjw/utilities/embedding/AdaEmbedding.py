from typing import Optional

import numpy as np
import torch

from cjw.utilities.embedding.Embedding import Embedding


class AdaEmbedding(Embedding):
    async def embed1(self, text: str) -> Optional[np.ndarray]:
        pass

    async def embed(self, text: str) -> Optional[np.ndarray]:
        ...
