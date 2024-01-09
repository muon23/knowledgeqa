from typing import List

from cjw.knowledgeqa.indexers.Indexer import Indexer


class PineconeIndexer(Indexer):
    @classmethod
    def new(cls, serverUrl: str, indexName: str, **kwargs) -> "PineconeIndexer":
        ...

    def __init__(self, serverUrl: str, indexName: str):
        ...

    async def add(self, data: List[dict], keyFields: List[str], idField: str = "_id", **kwargs):
        ...

    async def search(self, query: str, top: int = 3, **kwargs) -> List[dict]:
        ...

    async def get(self, ids: str | List[str]) -> List[dict]:
        ...

    async def delete(self, ids: str | List[str]):
        ...

    async def kill(self):
        ...

    async def size(self):
        ...