import logging
from typing import List

from cjw.knowledgeqa.evaluators.Evaluator import Evaluator
from cjw.knowledgeqa.evaluators.QAData import QAData
from cjw.knowledgeqa.indexers import Indexer


class ProximityEvaluator(Evaluator):

    logger = logging.getLogger(__qualname__)

    __DEFAULT_SCORES = [1.0, 0.5, 0.2]

    def __init__(self, scores: List[float] = None):
        super().__init__()
        self.testSet: QAData | None = None
        self.indexer: Indexer | None = None
        self.scores: List[float] = scores if scores else self.__DEFAULT_SCORES

    async def withData(self, testSet: QAData, indexer: Indexer, refreshIndex=False) -> "ProximityEvaluator":
        self.testSet = testSet
        self.indexer = indexer

        refreshIndex = refreshIndex or await indexer.size() == 0

        if refreshIndex:
            await self.indexer.add(self.testSet.to_dict(), keyFields=["answer"])
        return self

    async def evaluate(self, sampleSize: int = 0, **kwargs) -> float:
        if not self.bot:
            raise RuntimeError("Nothing to evaluate (need a bot)")

        if not self.testSet:
            raise RuntimeError("Need sample Q&A data")
        data = self.testSet.sample(sampleSize) if sampleSize else self.testSet

        positives = 0.0

        for qa in data.records():
            question = qa['question']
            self.logger.info(f"Question: {question}")

            answer = await self.bot.ask(question, **self.botArgs)
            self.logger.info(f"Answer: {answer}")

            if answer.citations[0] == "--":  # The bot doesn't know
                continue

            proximity = await self.indexer.search(answer.content, top=len(self.scores))
            self.logger.info(
                "".join([f"\n[{p['_id']}] {p['answer']}" for p in proximity])
            )

            try:
                standardAnswers = [p['_id'] for p in proximity]
                score = max([self.scores[standardAnswers.index(a)] for a in answer.citations])
                self.logger.info(f"Score = {score}")
                positives += score
            except ValueError:
                pass    # Not found in proximity so no score

        if sampleSize == 0:
            sampleSize = self.testSet.size()

        return positives / sampleSize




