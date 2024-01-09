from typing import List
from uuid import uuid4

import marqo

from cjw.knowledgeqa.indexers.Indexer import Indexer


class MarqoIndexer(Indexer):
    DEFAULT_BATCH_SIZE = 20

    @classmethod
    def new(cls, serverUrl: str, indexName: str, **kwargs) -> "MarqoIndexer":
        mq = marqo.Client(serverUrl)
        indices = [idx.index_name for idx in mq.get_indexes()["results"]]
        if indexName in indices:
            raise Indexer.IndexExistError(indexName)
        mq.create_index(indexName, **kwargs)

        return MarqoIndexer(serverUrl, indexName)

    def __init__(self, serverUrl: str, indexName: str):
        self.indexName = indexName
        self.server = marqo.Client(serverUrl)

        try:
            self.index = self.server.get_index(self.indexName)
        except marqo.errors.MarqoWebError:
            raise Indexer.IndexNotFoundError(self.indexName)

    async def add(self, data: List[dict], keyFields: List[str], idField: str = None, **kwargs):
        if not idField:
            for item in data:
                if "_id" not in item:
                    item["_id"] = uuid4()

        elif idField != "_id":
            for item in data:
                item["_id"] = item[idField]

        status = self.index.add_documents(
            data,
            tensor_fields=keyFields,
            auto_refresh=True,
            client_batch_size=self.DEFAULT_BATCH_SIZE,
            **kwargs
        )
        return status[0]

    async def search(self, query: str, top: int = 3, **kwargs) -> List[dict]:
        results = self.index.search(q=query, limit=top, **kwargs)
        return results["hits"]

    async def get(self, ids: str | List[str]) -> List[dict]:
        if isinstance(ids, str):
            ids = [ids]

        results = self.index.get_documents(document_ids=ids)
        return results["results"]

    async def delete(self, ids: str | List[str]):
        if isinstance(ids, str):
            ids = [ids]

        status = self.index.delete_documents(ids=ids)
        return status

    async def kill(self):
        status = self.server.delete_index(self.indexName)
        return status

    async def size(self):
        stats = self.index.get_stats()
        return stats['numberOfDocuments']
