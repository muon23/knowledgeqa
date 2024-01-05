class Evaluator:
    """To evaluate the effectiveness of the bots """
    def __init__(self, testSet: str, model: str, method: str):
        ...

    def llmMethod(self):
        """Ask AI how good the results are"""
        ...

    def embeddingMethod(self):
        """See if the correct answer is among the top picked embedding from the response"""
        ...

