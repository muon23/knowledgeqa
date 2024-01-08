import asyncio
import unittest

from cjw.knowledgeqa.bots.GptBot import GptBot
from cjw.knowledgeqa.indexer.MarqoIndexer import MarqoIndexer


class TestGptBot(unittest.TestCase):
    MARQO_SERVER = "http://localhost:8882"
    TEST_INDEX_NAME = "test_indexer"

    def test_basic(self):
        index = MarqoIndexer(self.MARQO_SERVER, self.TEST_INDEX_NAME)
        question = "Do you know any point of interests in Michigan?"

        loop = asyncio.get_event_loop()

        bot = GptBot.of("gpt4").withFacts(index, contentFields=["title", "text"], top=5)

        answer = loop.run_until_complete(bot.ask(question))
        print(question)
        print(f"{answer.content} [{answer.citation}]")

        unknown = "What is Euler identity?"
        answer2 = loop.run_until_complete(bot.ask(unknown))
        print(unknown)
        print(f"{answer2.content} [{answer2.citation}]")

        answer3 = loop.run_until_complete(bot.ask(unknown, restricted=False))
        print(unknown)
        print(f"{answer3.content} [{answer3.citation}]")

        bot.withFacts(None)
        answer4 = loop.run_until_complete(bot.ask(unknown))
        print(unknown)
        print(f"{answer4.content} [{answer4.citation}]")


if __name__ == '__main__':
    unittest.main()
