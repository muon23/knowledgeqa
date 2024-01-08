from cjw.knowledgeqa.evaluators.Evaluator import Evaluator


class SelfSimilarityEvaluator(Evaluator):

    async def evaluate(self, examples: int = 0, **kwargs):
        ...
