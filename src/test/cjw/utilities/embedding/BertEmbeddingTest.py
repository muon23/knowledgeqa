import unittest

from cjw.utilities.Languages import Languages
from cjw.utilities.embedding.BertEmbedding import BertEmbedding


class BertEmbeddingTest(unittest.TestCase):

    def test_encodeSentences(self):
        article = """
        预告：27日15时《李献计历险记》首映发布会>>>点击进入电影《李献计》首映发布会直播间(27日14时半开通)
        """
        bert = BertEmbedding("distilbert-multilingual-nli-stsb-quora-ranking")

        article = Languages.implySentences(article)
        print(">>>>", article)

        sentence, vectors = bert.embedSentences(article)

        print(sentence, vectors)

    def test_encode1(self):
        article = """
        预告：27日15时《李献计历险记》首映发布会>>>点击进入电影《李献计》首映发布会直播间(27日14时半开通)
        """
        bert = BertEmbedding("distilbert-multilingual-nli-stsb-quora-ranking")

        print(">>>>", article)

        vector = bert.embed1(article)

        print(vector)

    def test_mew_model(self):
        article = """
        预告：27日15时《李献计历险记》首映发布会>>>点击进入电影《李献计》首映发布会直播间(27日14时半开通)
        """
        bert = BertEmbedding("Multilingual-MiniLM-L12-H384")

        print(">>>>", article)

        vector = bert.embed1(article)

        print(vector)


if __name__ == '__main__':
    unittest.main()
