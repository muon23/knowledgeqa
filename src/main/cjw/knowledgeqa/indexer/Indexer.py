from cjw.utilities.db.Database import Database
from cjw.utilities.embedding.Embedding import Embedding


class Indexer:
    @classmethod
    def of(cls, embedding: str, database: str) -> "Indexer":
        ...

    def __init__(self, embedding: Embedding, database: Database):
        self.embedding = embedding
        self.database = database
