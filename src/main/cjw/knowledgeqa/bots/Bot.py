from abc import ABC, abstractmethod

from cjw.knowledgeqa.bots.Answer import Answer


class Bot(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def ask(self, question: str) -> Answer:
        pass

