import asyncio
import os
import unittest

from cjw.knowledgeqa.bots import Bot
from cjw.utilities.llm.ChatPrompt import ChatPrompt
from cjw.utilities.llm.MistralPortal import MistralPortal


class MistralPortalTest(unittest.TestCase):
    def test2(self):
        bot = Bot()
        result = bot.ask("This won't do a thing")
        print(result)

    def test_basic(self):
        key = os.environ["MISTRAL_API_KEY"]
        mistral = MistralPortal.of(key=key)

        prompt = ChatPrompt(bot="assistant")
        prompt.user("What models does MistralAI support?")

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mistral.chatCompletion(messages=prompt.messages))

        print(result)

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
