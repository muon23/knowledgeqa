from abc import ABC, abstractmethod
from typing import Optional

from cjw.knowledgeqa.bots.Bot import Bot


class Evaluator(ABC):
    def __init__(self):
        self.bot: Optional[Bot] = None
        self.botArgs = dict()

    def forBot(self, bot: Bot, **kwarg) -> "Evaluator":
        self.bot = bot
        self.botArgs = kwarg
        return self

    @abstractmethod
    async def evaluate(self, sampleSize: int = 0, **kwargs) -> float:
        pass

