from typing import List

from cjw.knowledgeqa.indexer.Indexer import Indexer


class MarqoIndexer(Indexer):
    @classmethod
    def of(cls, serverUrl: str, indexName: str, **kwargs) -> "MarqoIndexer":
        ...

    async def add(self, data: List[dict], keyFields: List[str], idField: str = "_id"):
        ...

    async def query(self, query: str) -> dict:
        ...
