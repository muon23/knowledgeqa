from cjw.knowledgeqa.indexers.Indexer import Indexer
from cjw.knowledgeqa.indexers.MarqoIndexer import MarqoIndexer
from cjw.knowledgeqa.indexers.PineconeIndexer import PineconeIndexer


def index(method: str, new=False, **kwargs) -> Indexer:
    if method.lower() == "marqo":
        if new:
            try:
                return MarqoIndexer.new(**kwargs)
            except Indexer.IndexExistError:
                # Exists, but OK, we will open the existing one
                pass
        return MarqoIndexer(**kwargs)

    elif method.lower() == "pinecone":
        return PineconeIndexer(**kwargs)

    else:
        raise NotImplementedError(f"Indexer {method} not supported")
