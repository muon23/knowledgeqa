import asyncio
import logging
import random
from typing import List

import numpy as np

from cjw.knowledgeqa.evaluators.Evaluator import Evaluator
from cjw.utilities.embedding.Embedding import Embedding


class ConsistencyEvaluator(Evaluator):

    logger = logging.getLogger(__qualname__)

    def __init__(self, sampleQuestions: List[str], embedding: Embedding):
        super().__init__()
        self.__sampleQuestions = sampleQuestions
        self.__embedding = embedding

    @classmethod
    def _similarity(cls, a: np.ndarray, b: np.ndarray) -> float:
        # Calculate cosine similarity between two vectors
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    @classmethod
    def _pairwiseSimilarity(cls, vectors: np.ndarray) -> np.ndarray:
        # Calculate cosine similarities between each pair of multiple vectors
        pairs = np.stack(np.triu_indices(len(vectors), k=1), axis=1)
        return np.array([cls._similarity(t[0], t[1]) for t in vectors[pairs]])

    async def _question2embedding(self, question: str) -> np.ndarray:
        # Ask the bot a question, and then embed its answer
        answer = await self._bot.ask(question)
        self.logger.info(f"Answer: {answer}")
        return await self.__embedding.embed(answer.content)

    async def evaluate(self, sampleSize: int = 0, tries: int = 3) -> float:
        """Evaluate the :class:`Bot`

        Args:
            sampleSize (int): How many test cases to run through the bot.  Zero means running with all provided test data.
            tries (int): Ask the same question this many times to the bot in order to check the consistency

        Returns:
            Performance score of the Bot
        """
        if not self._bot:
            raise RuntimeError("Nothing to evaluate (need a bot)")

        if not self.__embedding:
            raise RuntimeError("No embedding model provided")

        # Pick the questions
        questions = random.sample(self.__sampleQuestions, sampleSize) if sampleSize else self.__sampleQuestions

        score = 0.0
        for q in questions:
            # For each question, ask the bot many times in parallel, and gather all the results
            embeddings = asyncio.gather(*[self._question2embedding(q) for _ in range(tries)])

            # Get the pairwise similarities.  Then, take the mean as the score.
            similarity = self._pairwiseSimilarity(np.array(await embeddings))
            score += np.mean(similarity)

        # Calculate the final score
        if sampleSize == 0:
            sampleSize = len(self.__sampleQuestions)

        return score / sampleSize
