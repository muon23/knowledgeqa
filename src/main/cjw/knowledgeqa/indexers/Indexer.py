from abc import ABC, abstractmethod
from typing import List, Any


class Indexer(ABC):
    """The base class of an Indexer, which utilizes vector database to search for relevant articles in the embedding
    space."""

    # Exceptions
    class IndexExistError(Exception):
        def __init__(self, indexName: str):
            super().__init__(f"Index {indexName} exists")

    class IndexNotFoundError(Exception):
        def __init__(self, indexName: str):
            super().__init__(f"Index {indexName} not found")

    @abstractmethod
    async def add(self, data: List[dict], keyFields: List[str], idField: str = "_id", **kwargs) -> Any:
        """ Adds data to the index.

        Args:
            data (List[dict]): The data
            keyFields (List[str]): Filed names whose values are used in embedding
            idField (str): The field name for the document ID.  Existing IDs in the index will be replaced.  (default "_id")
            **kwargs: Other subclass specific arguments

        Returns:
            Subclass specific status
        """
        # TODO: use pandas DataFrame for data
        pass

    @abstractmethod
    async def search(self, query: str, top: int = 3, **kwargs) -> List[dict]:
        """ Searches for nearest neighbors of the query in the embedding space.

        Args:
            query (str): The text to fish out relevant documents in the database
            top (int): Number of documents to get
            **kwargs: Other subclass specific parameters

        Returns:
            The documents with their meta data
        """
        pass

    @abstractmethod
    async def get(self, ids: str | List[str]) -> List[dict]:
        """ Gets specific documents by their IDs

        Args:
            ids (str | List[str]): The ID(s)

        Returns:
            The documents
        """
        pass

    @abstractmethod
    async def delete(self, ids: str | List[str]) -> Any:
        """ Deletes specific documents by their IDs

        Args:
            ids (str | List[str]): The ID(s)

        Returns:
            Subclass specific status
        """
        pass

    @abstractmethod
    async def kill(self) -> Any:
        """ Removes the index from the database.  All data are lost.

        Returns:
            Subclass specific status
        """
        pass

    @abstractmethod
    async def size(self) -> Any:
        """Returns number of documents in the index."""
        pass
