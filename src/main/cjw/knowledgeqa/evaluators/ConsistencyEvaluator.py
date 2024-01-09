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
        self.sampleQuestions = sampleQuestions
        self.embedding = embedding

    @classmethod
    def similarity(cls, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    @classmethod
    def pairwiseSimilarity(cls, vectors: np.ndarray) -> np.ndarray:
        pairs = np.stack(np.triu_indices(len(vectors), k=1), axis=1)
        return np.array([cls.similarity(t[0], t[1]) for t in vectors[pairs]])

    async def question2embedding(self, question: str) -> np.ndarray:
        answer = await self.bot.ask(question)
        self.logger.info(f"Answer: {answer}")
        return await self.embedding.embed(answer.content)

    async def evaluate(self, sampleSize: int = 0, tries: int = 3, **kwargs) -> float:
        if not self.bot:
            raise RuntimeError("Nothing to evaluate (need a bot)")

        if not self.embedding:
            raise RuntimeError("No embedding model provided")

        questions = random.sample(self.sampleQuestions, sampleSize) if sampleSize else self.sampleQuestions

        score = 0.0
        for q in questions:
            embeddings = asyncio.gather(*[self.question2embedding(q) for _ in range(tries)])
            similarity = self.pairwiseSimilarity(np.array(await embeddings))
            score += np.mean(similarity)

        if sampleSize == 0:
            sampleSize = len(self.sampleQuestions)

        return score / sampleSize
