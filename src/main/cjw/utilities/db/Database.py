from abc import ABC, abstractmethod


class Database(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def upsert(self, key, value):
        pass

    @abstractmethod
    async def query(self, key):
        pass
