import logging
import numpy as np
import transformers as tf
from typing import Tuple, Optional

from cjw.utilities.Languages import Languages
from cjw.utilities.embedding.Embedding import Embedding


#
# Known issues:
# - It seemed that you cannot create multiple BertEncoder at the same time.  Something in the BERT library would
#   overwrite earlier loaded models with the last one.
#

class BertEmbedding(Embedding):
    """
    Uses BERT to encode a passage of text into vectors.
    """

    # TODO: Set HF_HOME to specify model location

    modelSpec = {
        # [ tokenizer class, model class, path to model in HuggingFace, languages applied ]
        "bert-base-chinese":
            [tf.AutoTokenizer, tf.AutoModelWithLMHead, None, ["zh"]],
        "distilbert-base-uncased":
            [tf.DistilBertTokenizer, tf.DistilBertModel, None, ["en"]],
        "bert-base-uncased":
            [tf.BertTokenizer, tf.BertModel, None, ["en"]],
        "distilbert-base-multilingual-cased":
            [tf.AutoTokenizer, tf.AutoModel, None, ["zh", "en"]],
        "roberta-large-mnli":
            [tf.AutoTokenizer, tf.AutoModel, None, ["en"]],
        "MiniLM-L12-H384-uncased":
            [tf.AutoTokenizer, tf.AutoModel, "microsoft", ["en"]],
        "xlm-r-100langs-bert-base-nli-stsb-mean-tokens":
            [tf.AutoTokenizer, tf.AutoModel, "sentence-transformers", ["zh", "en"]],
        "xlm-r-100langs-bert-base-nli-mean-tokens":
            [tf.AutoTokenizer, tf.AutoModel, "sentence-transformers", ["zh", "en"]],
        "distilbert-multilingual-nli-stsb-quora-ranking":
            [tf.AutoTokenizer, tf.AutoModel, "sentence-transformers", ["zh", "en"]],
        "Multilingual-MiniLM-L12-H384":
            [tf.XLMRobertaTokenizer, tf.AutoModel, "microsoft", ["zh", "en"]],
        "DialoGPT-large":
            [tf.AutoTokenizer, tf.AutoModel, "microsoft", ["en"]],
        "DialoGPT-small":
            [tf.AutoTokenizer, tf.AutoModel, "microsoft", ["en"]],
    }

    def __init__(self, modelName):
        if modelName not in self.modelSpec:
            raise ValueError(f"unsupported model name {modelName}")

        tokenizer, model, path, self.languages = self.modelSpec[modelName]
        if path:
            modelName = path + "/" + modelName

        self.tokenizer = tokenizer.from_pretrained(modelName)

        self.padToken = modelName in ["DialoGPT-large", "DialoGPT-small"]
        if self.padToken:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = model.from_pretrained(modelName)
        logging.info(f"BERT model {modelName} loaded")

    def embedSentences(self, text: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Breaks the text into sentences, and encodes each one into a vector with BERT.
        It also separates English sentences from Chinese, and apply appropriately to the BERT model that support
        the language.  If a model only supports English, all Chinese sentences are ignored, and vise versa.
        :param text: The text to be encoded
        :return: A numpy array of sentence vectors
        """
        ideoText, alphaText = Languages.separateIdeograph(text)
        hanSent = Languages.hanziSentences(ideoText)
        engSent = Languages.englishSentences(alphaText)

        sentences = (hanSent if "zh" in self.languages else []) + \
                    (engSent if "en" in self.languages else [])

        if len(sentences) == 0:
            return None, None

        tokenized = self.tokenizer(
            sentences,
            padding=True, max_length=512, truncation=True,
            add_special_tokens=self.padToken,
            return_tensors='pt'
        )

        # with torch.no_grad():
        #     lastHiddenStates = self.model(**tokenized)
        lastHiddenStates = self.model(**tokenized)

        return sentences, lastHiddenStates[0][:, 0, :].detach().numpy()

    async def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a text into a vector with BERT.
        The text is broken into sentences, and the average of the vectors for the sentences are used.
        :param text: The text to be encoded.
        :return: A vector representing the text.  None for if the text is empty.
        """
        sentences, vectors = self.embedSentences(text)
        return np.mean(vectors, axis=0) if sentences else None

    def embed1(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text without breaking it to sentences first.  If the text is too long for BERT (512 tokens), it is will
        be truncated.  This is more efficient than encode() if one knows the text will not be long.
        :param text: The text to be encoded.
        :return: A vector representing the text.
        """
        tokenized = self.tokenizer(
            [text],
            padding=True, max_length=512, truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        lastHiddenStates = self.model(**tokenized)

        return lastHiddenStates[0][:, 0, :].detach().numpy()[0]

