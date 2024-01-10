import logging
from typing import List, Optional

from cjw.knowledgeqa.evaluators.Evaluator import Evaluator
from cjw.knowledgeqa.evaluators.QAData import QAData
from cjw.knowledgeqa.indexers import Indexer


class ProximityEvaluator(Evaluator):
    """ Evaluate a :class:`Bot` by checking if the standard answers is in the proximity of its answer."""

    logger = logging.getLogger(__qualname__)

    __DEFAULT_SCORE_WEIGHTS = [1.0, 0.5, 0.2]  # Default weights of the top-proximity standard answers

    def __init__(self, scores: List[float] = None):
        """ The constructor.

        Args:
            scores (List[float]): Scores assigned to the answer from the :class:Bot if the standard answer is among the top searches using the answer.  For example, the default is [1.0, 0.5, 0.2], meaning if the standard answer is the 2nd closest to the answer, it is scored 0.5.
        """
        super().__init__()
        self.__testSet: Optional[QAData] = None
        self.__indexer: Optional[Indexer] = None
        self.__scoreWeights: List[float] = scores if scores else self.__DEFAULT_SCORE_WEIGHTS

    async def withData(self, testSet: QAData, indexer: Indexer, refreshIndex=False) -> "ProximityEvaluator":
        """ Attaches a test set.

        Args:
            testSet (QAData): A set of questions and standard answers.
            indexer (Indexer): An :class:`Indexer` to hold and search the standard answers.
            refreshIndex (bool): True if we want to refresh the index with data in the testSet.  (default False)

        Returns:
            The ProximityEvaluator itself.
        """
        self.__testSet = testSet
        self.__indexer = indexer

        # If nothing is in the index, we automatically add the Q&A to the index
        refreshIndex = refreshIndex or await indexer.size() == 0

        # Add the test data to the index if necessary
        if refreshIndex:
            await self.__indexer.add(self.__testSet.to_dict(), keyFields=["answer"])
        return self

    async def evaluate(self, sampleSize: int = 0, showFailedQuestions=False) -> float:
        """Evaluate the :class:`Bot`

        Args:
            sampleSize (int): How many test cases to run through the bot.  Zero means running with all provided test data.
            showFailedQuestions (bool): Print failed questions for debugging. (default False)

        Returns:
            Performance score of the Bot
        """
        if not self._bot:
            raise RuntimeError("Nothing to evaluate (need a bot)")

        if not self.__testSet:
            raise RuntimeError("Need sample Q&A data")

        # Select test data
        data = self.__testSet.sample(sampleSize) if sampleSize else self.__testSet

        # Iterate over the test data
        positives = 0.0
        for qa in data.records():
            question = qa['question']
            self.logger.info(f"Question: {question}")

            # Ask the bot a question
            answer = await self._bot.ask(question, **self._botArgs)
            self.logger.info(f"Answer: {answer}")

            if answer.citations[0] == "--":  # The bot doesn't know, so no score
                if showFailedQuestions:
                    print(f"Failed question: {question} ({answer})")
                continue

            # Search the index for similar answers
            proximity = await self.__indexer.search(answer.content, top=len(self.__scoreWeights))
            self.logger.info(
                "".join([f"\n[{p['_id']}] {p['answer']}" for p in proximity])
            )

            try:
                # If any of the standard questions found in the index matches the question given, score this round.
                testQuestions = [p['question'] for p in proximity]
                score = self.__scoreWeights[testQuestions.index(question)]
                self.logger.info(f"Score = {score}")
                positives += score
            except ValueError:
                # The question given is not amongst the standard ones.  Scores 0.
                if showFailedQuestions:
                    print(f"Failed question: {question} ({answer})")
                pass    # Not found in proximity so no score

        if sampleSize == 0:
            sampleSize = self.__testSet.size()

        return positives / sampleSize




