from abc import ABC, abstractmethod
from typing import List


class Indexer(ABC):

    @abstractmethod
    async def add(self, data: List[dict], keyFields: List[str], idField: str = "_id"):
        pass

    @abstractmethod
    async def query(self, query: str) -> dict:
        pass
