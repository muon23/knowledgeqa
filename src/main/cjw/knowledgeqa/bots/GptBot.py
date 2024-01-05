from cjw.knowledgeqa.bots.Answer import Answer
from cjw.knowledgeqa.bots.Bot import Bot


class GptBot(Bot):
    async def ask(self, question: str) -> Answer:
        ...
