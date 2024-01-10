import logging
import os
from typing import TypeVar, Optional

from cjw.knowledgeqa.bots.Answer import Answer
from cjw.knowledgeqa.bots.Bot import Bot
from cjw.utilities.llm.ChatPrompt import ChatPrompt
from cjw.utilities.llm.GptPortal import GptPortal


class GptBot(Bot):
    """A :class:`Bot` running OpenAI models."""

    GptBot = TypeVar("GptBot")

    logger = logging.getLogger(__qualname__)

    __MODEL_TRANSLATION = {
        "gpt3": "text-davinci-003",
        "gpt3.5": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
        "gpt4+": "gpt-4-1106-preview",
    }

    models = set(list(__MODEL_TRANSLATION.keys()) + list(__MODEL_TRANSLATION.values()))  # All supported model names

    #
    # Prompts
    #
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

    DEFAULT_MAX_TOKENS = 6000   # Limit of prompt tokens.  RAG facts are added to the prompt until this limit exceeds.
    DEFAULT_TEMPERATURE = 0.7   # Creativity of the Bot

    @classmethod
    def of(cls, model: str = None, key: str = None, **kwargs: object) -> GptBot:
        """ A friendier way to create GptBot

        Args:
            model (str): The model name, or the supported short-hand of it.
            key (str): The OpenAI API key
            **kwargs: Additional OpenAI connection arguments. See :class:`GptPortal`.

        Returns:
            A GptBot
        """
        if model is None:
            return GptBot(model="default", **kwargs).withKey(key)
        elif model.lower() in cls.__MODEL_TRANSLATION:
            return GptBot(model=cls.__MODEL_TRANSLATION[model], **kwargs).withKey(key)
        elif model.lower() in cls.__MODEL_TRANSLATION.values():
            return GptBot(model=model, **kwargs).withKey(key)
        else:
            return None

    def __init__(self, model: Optional[str], **kwargs):
        """ The constructor

        Args:
            model (str): Name of the OpenAI model.
            **kwargs: Additional OpenAI connection arguments. See :class:`GptPortal`.
        """
        super().__init__(**kwargs)

        self.model = model      # The model name
        self._portal: Optional[GptPortal] = None  # Handle communication to OpenAI's GPT

    def __eq__(self, other):
        """True if two GptBots using the same model."""
        return (
                isinstance(other, GptBot) and
                super().__eq__(other) and
                self.model == other.model
        )

    def withKey(self, key: str = None) -> "GptBot":
        """Replaces the OpenAI API key.

        Args:
            key (str): The OpenAI API key

        Returns:
            The GptBot itself.
        """
        accessKey = key if key else os.environ.get("OPENAI_API_KEY")
        if not accessKey:
            raise RuntimeError("No access key was given")

        self._portal = GptPortal.of(accessKey)
        return self

    async def ask(self, question: str, **kwargs) -> Answer:
        """ Ask GptBot a question.

        Args:
            question (str): The question
            **kwargs:
                - restricted (bool): True if the bot shall answer strictly from the given fact. (default True)
                - idField (str): Which field of the indexed facts is used for citation. (default "_id")
                - contentFields (List[str]): Which fields of the indexed facts are used to construct the prompt.  (default ["text"])
                - maxToken (int): Max number of tokens for the prompt. (default DEFAULT_MAX_TOKENS)
                - temperature (float): How creative the Bot should be. (default DEFAULT_TEMPERATURE)

        Returns:
            The :class:`Answer`.
        """
        if not self._portal:
            self.withKey()

        prompt = ChatPrompt(bot="assistant")    # GPT uses "assistant" to identify the AI

        # Construct the instruction prompt
        restricted = kwargs.pop("restricted", True)
        if self._indexer:
            instruction = self.__INSTRUCTION_WITH_FACTS_ONLY if restricted else self.__INSTRUCTION_WITH_UPDATES
        else:
            instruction = self.__INSTRUCTION

        prompt.system(instruction)
        numTokens = self._portal.estimateTokens(instruction)

        # If RAG is used, construct the prompt with known facts.
        if self._indexer:
            searchArgs = self._indexerArgs.copy()
            idField = searchArgs.pop("idField", "_id")
            contentFields = searchArgs.pop("contentFields", ["text"])
            maxTokens = searchArgs.pop("maxTokens", self.DEFAULT_MAX_TOKENS)

            # Search the index in the embedding space for the semantically closed facts
            found = await self._indexer.search(question, **searchArgs)

            # Put the facts together into a prompt.  The result of the search shall be sorted by their similarities.
            facts = dict()
            for f in found:
                content = " -- ".join([f.get(c, "") for c in contentFields])    # Join all content fields together
                numTokens += self._portal.estimateTokens(content)   # Count the token
                if numTokens > maxTokens:
                    # Too many tokens.  Ignore some low-score facts
                    break

                facts[f[idField]] = content     # Added to the collection of relevant facts

            # Join all the facts together with their citations and add them to the prompt
            gatheredFacts = "\n\n".join([f"[{fid}]\n{facts[fid]}" for fid in facts])
            self.logger.info(f"Considering facts:\n{gatheredFacts}")
            prompt.system(gatheredFacts, replace=False)

        # Add user question to the prompt
        prompt.user(question)

        # Invoke chat_completion API of GPT
        if "temperature" not in kwargs:
            kwargs["temperature"] = self.DEFAULT_TEMPERATURE

        self.logger.info(f"Question:\n{question}")
        responses = await self._portal.chatCompletion(prompt.messages, **kwargs)
        self.logger.info(f"Answer:\n{responses['content']}")

        return Answer.of(responses["content"])



