from abc import ABC, abstractmethod
from typing import Optional

from cjw.knowledgeqa.bots.Bot import Bot


class Evaluator(ABC):
    """Base class for evaluators that evaluates the performance of :class:`Bot` objects."""
    def __init__(self):
        self._bot: Optional[Bot] = None
        self._botArgs = dict()

    def forBot(self, bot: Bot, **kwarg) -> "Evaluator":
        """Attaches a Bot to evaluate

        Args:
            bot (Bot): The bot to evaluate
            **kwarg: Arguments when asking questions to the Bots

        Returns:
            The Evaluator itself.
        """
        self._bot = bot
        self._botArgs = kwarg
        return self

    @abstractmethod
    async def evaluate(self, sampleSize: int = 0, **kwargs) -> float:
        """ Evaluate the :class:`Bot`

        Args:
            sampleSize (int): How many test cases to run through the bot.  Zero means running with all provided test data.
            **kwargs: Additional subclass specific arguments

        Returns:
            Performance score of the :class:`Bot`
        """
        pass

