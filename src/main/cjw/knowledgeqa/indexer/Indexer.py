from abc import ABC, abstractmethod
from typing import List


class Indexer(ABC):
    class IndexExistError(Exception):
        def __init__(self, indexName: str):
            super().__init__(f"Index {indexName} exists")

    class IndexNotFoundError(Exception):
        def __init__(self, indexName: str):
            super().__init__(f"Index {indexName} not found")

    @abstractmethod
    async def add(self, data: List[dict], keyFields: List[str], idField: str = "_id", **kwargs):
        pass

    @abstractmethod
    async def search(self, query: str, top: int = 3, **kwargs) -> List[dict]:
        pass

    @abstractmethod
    async def get(self, ids: str | List[str]) -> List[dict]:
        pass

    @abstractmethod
    async def delete(self, ids: str | List[str]):
        pass

    @abstractmethod
    async def kill(self):
        pass
