from cjw.knowledgeqa.evaluators.Evaluator import Evaluator


class BotAssistedEvaluator(Evaluator):

    async def evaluate(self, examples: int = 0, **kwargs):
        ...
