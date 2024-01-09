from abc import ABC, abstractmethod

from cjw.knowledgeqa.bots.Bot import Bot


class Evaluator(ABC):
    def __init__(self):
        self.bot: Bot | None = None
        self.botArgs = dict()

    def forBot(self, bot: Bot, **kwarg) -> "Evaluator":
        self.bot = bot
        self.botArgs = kwarg
        return self

    @abstractmethod
    async def evaluate(self, sampleSize: int = 0, **kwargs) -> float:
        pass

