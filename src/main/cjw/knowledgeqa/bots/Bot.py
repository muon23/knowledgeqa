from abc import ABC, abstractmethod

from cjw.knowledgeqa.bots.Answer import Answer
from cjw.knowledgeqa.indexers.Indexer import Indexer


class Bot(ABC):
    def __init__(self, **kwargs):
        self.indexer: Indexer | None = None
        self.indexerArgs: dict = dict()

    def withFacts(
            self,
            indexer: Indexer = None,
            **kwargs
    ) -> "Bot":
        self.indexer = indexer
        self.indexerArgs = kwargs
        return self

    @abstractmethod
    async def ask(self, question: str, facts: dict = None, **kwargs) -> Answer:
        pass

