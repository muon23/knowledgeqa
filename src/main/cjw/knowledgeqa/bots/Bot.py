from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from cjw.knowledgeqa.bots.Answer import Answer
from cjw.knowledgeqa.indexers.Indexer import Indexer


class Bot(ABC):
    """The base class of a Bot that answers questions."""

    def __init__(self, **kwargs):
        self._indexer: Optional[Indexer] = None
        self._indexerArgs: Dict[str, Any] = dict()

    def withFacts(
            self,
            indexer: Indexer = None,
            **kwargs
    ) -> "Bot":
        """Adds known facts to the bot for Retrival-Augmented Generation (RAG)

        Args:
            indexer (Indexer): The indexed facts.  See :class:`Indexer`.
            **kwargs: Arguments to give the indexer during the fact searching.   See :meth:`Indexer.search`

        Returns:
            The Bot itself.
        """
        self._indexer = indexer
        self._indexerArgs = kwargs
        return self

    @abstractmethod
    async def ask(self, question: str, **kwargs) -> Answer:
        """  Ask the bot a question.

        Args:
            question: Any question for the Bot
            **kwargs: Additional arguments for the specific LLM model.

        Returns:
            The :class:`Answer`.
        """
        pass

