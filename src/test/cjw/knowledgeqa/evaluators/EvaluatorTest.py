import asyncio
import logging
import unittest

from cjw.knowledgeqa import indexers, bots
from cjw.knowledgeqa.bots.Bot import Bot
from cjw.knowledgeqa.evaluators.ConsistencyEvaluator import ConsistencyEvaluator
from cjw.knowledgeqa.evaluators.ProximityEvaluator import ProximityEvaluator
from cjw.knowledgeqa.evaluators.QAData import QAData
from cjw.knowledgeqa.indexers import Indexer
from cjw.utilities.embedding.BertEmbedding import BertEmbedding


class EvaluatorTest(unittest.TestCase):
    DATA_FILE = "../../../../../data/wikipedia_question_similar_answer.tsv"
    MARQO_SERVER = 'http://localhost:8882'
    TEST_INDEX_NAME = "wiki_test_qa"
    RAG_KNOWLEDGE = 5

    data: QAData = None
    index: Indexer = None
    bot: Bot = None

    def setUp(self) -> None:
        self.data = QAData(self.DATA_FILE)
        self.index = indexers.index("marqo", new=True, serverUrl=self.MARQO_SERVER, indexName=self.TEST_INDEX_NAME)
        self.bot = bots.bot("gpt4").withFacts(self.index, contentFields=["answer"], top=self.RAG_KNOWLEDGE)

    def test_proximity(self):
        logging.basicConfig(level=logging.INFO)
        # ProximityEvaluator.logger.setLevel(logging.INFO)
        # GptBot.logger.setLevel(logging.INFO)

        loop = asyncio.get_event_loop()

        evaluator = ProximityEvaluator().forBot(self.bot)

        loop.run_until_complete(evaluator.withData(self.data, self.index))
        score = loop.run_until_complete(evaluator.evaluate(sampleSize=5, showFailedQuestions=True))

        print(score)

    def test_consistency(self):
        logging.basicConfig(level=logging.DEBUG)
        # ConsistencyEvaluator.logger.setLevel(logging.INFO)
        # GptBot.logger.setLevel(logging.INFO)

        loop = asyncio.get_event_loop()

        questions = self.data.getQuestions()
        bert = BertEmbedding("distilbert-multilingual-nli-stsb-quora-ranking")
        evaluator = ConsistencyEvaluator(questions, bert).forBot(self.bot)

        score = loop.run_until_complete(evaluator.evaluate(sampleSize=2))

        print(score)


if __name__ == '__main__':
    unittest.main()
