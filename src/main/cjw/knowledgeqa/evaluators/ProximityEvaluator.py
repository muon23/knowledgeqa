from cjw.knowledgeqa.evaluators.Evaluator import Evaluator
from cjw.knowledgeqa.evaluators.QAData import QAData
from cjw.knowledgeqa.indexers import Indexer


class ProximityEvaluator(Evaluator):

    def __init__(self, neighbors: int = 3):
        super().__init__()
        self.testSet: QAData | None = None
        self.indexer: Indexer | None = None
        self.neighbors: int = neighbors

    async def withData(self, testSet: QAData, indexer: Indexer) -> "ProximityEvaluator":
        self.testSet = testSet
        self.indexer = indexer
        await self.indexer.add(self.testSet.to_dict(), keyFields=["answer"])
        return self

    async def evaluate(self, sampleSize: int = 0, **kwargs) -> float:
        if not self.bot:
            raise RuntimeError("Nothing to evaluate (need a bot)")

        if not self.testSet:
            raise RuntimeError("Need sample Q&A data")

        data = self.testSet.sample(sampleSize) if sampleSize else self.testSet
        positives = 0

        for qa in data.records():
            answer = await self.bot.ask(qa['question'], **self.botArgs)

            proximity = await self.indexer.search(answer.content, top=self.neighbors)
            positives += 1 if answer.citation in [p['_id'] for p in proximity] else 0

        if sampleSize == 0:
            sampleSize = self.testSet.size()

        return positives / sampleSize




