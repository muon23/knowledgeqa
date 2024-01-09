import logging
import os
from typing import TypeVar

from cjw.knowledgeqa.bots.Answer import Answer
from cjw.knowledgeqa.bots.Bot import Bot
from cjw.utilities.llm.ChatPrompt import ChatPrompt
from cjw.utilities.llm.GptPortal import GptPortal


class GptBot(Bot):
    GptBot = TypeVar("GptBot")

    logger = logging.getLogger(__qualname__)

    __MODEL_TRANSLATION = {
        "gpt3": "text-davinci-003",
        "gpt3.5": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
    }

    models = set(list(__MODEL_TRANSLATION.keys()) + list(__MODEL_TRANSLATION.values()))

    ___EXAMPLES = """
    For example:
    
    Facts:
    [12345a]
    Pumps operate by some mechanism (typically reciprocating or rotary ), and consume energy to perform mechanical work by moving the fluid.
    ...
    
    
    User:
        How do I remove water from my basement?
    
    Assistant:
        You may try to use a pump.  [12345a]
    """

    __INSTRUCTION_WITH_FACTS_ONLY = f"""
    You will answer the user's questions based solly on the following facts.
    Each fact are preceded by an ID in square brackets in a separate line before the fact.
    Answer the questions by citing the ID in square brackets after your answer.
    Say "I don't know" and cite "[--]" if nothing can be found from the facts.
    
    {___EXAMPLES}
    """

    __INSTRUCTION_WITH_UPDATES = f"""
    You will answer the user's questions with updates information from the following facts.
    If you can't find the answer from the provided facts, answer it based on your best knowledge. 
    Each fact are preceded by an ID in square brackets in a separate line before the fact.
    Answer the questions by citing the ID in square brackets if your answer is derived from the provided facts.
    
    {___EXAMPLES}
    """

    __INSTRUCTION = "You will answer user's questions based on your best knowledge."

    DEFAULT_MAX_TOKENS = 6000

    @classmethod
    def of(cls, model: str = None, key: str = None, **kwargs: object) -> GptBot:
        if model is None:
            return GptBot(model="default", **kwargs).withKey(key)
        elif model.lower() in cls.__MODEL_TRANSLATION:
            return GptBot(model=cls.__MODEL_TRANSLATION[model], **kwargs).withKey(key)
        elif model.lower() in cls.__MODEL_TRANSLATION.values():
            return GptBot(model=model, **kwargs).withKey(key)
        else:
            return None

    def __init__(self, model: str | None, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.portal: GptPortal | None = None

    def __eq__(self, other):
        return (
                isinstance(other, GptBot) and
                super().__eq__(other) and
                self.model == other.model
        )

    def withKey(self, key: str = None) -> "GptBot":
        accessKey = key if key else os.environ.get("OPENAI_API_KEY")
        if not accessKey:
            raise RuntimeError("No access key was given")
        self.portal = GptPortal.of(accessKey)
        return self

    async def ask(self, question: str, **kwargs) -> Answer:
        if not self.portal:
            self.withKey()

        prompt = ChatPrompt(bot="assistant")

        restricted = kwargs.pop("restricted", True)
        if self.indexer:
            instruction = self.__INSTRUCTION_WITH_FACTS_ONLY if restricted else self.__INSTRUCTION_WITH_UPDATES
        else:
            instruction = self.__INSTRUCTION

        prompt.system(instruction)
        numTokens = self.portal.estimateTokens(instruction)

        if self.indexer:
            searchArgs = self.indexerArgs.copy()
            idField = searchArgs.pop("idField", "_id")
            contentFields = searchArgs.pop("contentFields", ["text"])
            maxTokens = searchArgs.pop("maxTokens", self.DEFAULT_MAX_TOKENS)

            found = await self.indexer.search(question, **searchArgs)

            facts = dict()
            for f in found:
                content = " -- ".join([f.get(c, "") for c in contentFields])
                numTokens += self.portal.estimateTokens(content)
                if numTokens > maxTokens:
                    # Too many tokens.  Ignore some low-score facts
                    break

                facts[f[idField]] = content

            gatheredFacts = "\n\n".join([f"[{fid}]\n{facts[fid]}" for fid in facts])
            self.logger.info(f"Considering facts:\n{gatheredFacts}")
            prompt.system(gatheredFacts, replace=False)

        prompt.user(question)

        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.7

        self.logger.info(f"Question:\n{question}")
        responses = await self.portal.chatCompletion(prompt.messages, **kwargs)
        self.logger.info(f"Answer:\n{responses['content']}")

        return Answer.of(responses["content"])



