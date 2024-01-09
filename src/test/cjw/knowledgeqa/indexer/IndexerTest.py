import asyncio
import json
import unittest
from typing import List

from cjw.knowledgeqa.indexers.Indexer import Indexer
from cjw.knowledgeqa.indexers.MarqoIndexer import MarqoIndexer


class IndexerTest(unittest.TestCase):
    TEST_DATA_DIR = "../../../../../data"
    TEST_DATA1 = f"{TEST_DATA_DIR}/simple.json"
    MARQO_SERVER = "http://localhost:8882"
    TEST_INDEX_NAME = "test_indexer"

    @classmethod
    def showMarqoResults(cls, results: List[dict]):
        for r in results:
            print(f"id={r['_id']} score={r['_score']} title={r['title']}\n{r['_highlights']}\n")

    def test_marqo_populate(self):
        loop = asyncio.get_event_loop()

        try:
            # Clean up the database
            loop.run_until_complete(MarqoIndexer(self.MARQO_SERVER, self.TEST_INDEX_NAME).kill())
        except Exception:
            pass

        try:
            # Shouldn't exist
            MarqoIndexer(self.MARQO_SERVER, self.TEST_INDEX_NAME)
            self.fail("The index should not exist")
        except Indexer.IndexNotFoundError as e:
            print(f"Raised exception as expected: {e}")

        # Create a new index
        index = MarqoIndexer.new(self.MARQO_SERVER, self.TEST_INDEX_NAME)

        try:
            MarqoIndexer.new(self.MARQO_SERVER, self.TEST_INDEX_NAME)
            self.fail("Should have raised index exist exception.")
        except Indexer.IndexExistError as e:
            print(f"Raised exception as expected: {e}")

        # Insert test data
        with open(self.TEST_DATA1, "r") as fd:
            data = json.load(fd)

        status = loop.run_until_complete(index.add(data, keyFields=["title", "text"], idField="id"))
        self.assertFalse(status["errors"])

        results = loop.run_until_complete(index.search("What is M-137?"))
        self.showMarqoResults(results)

        self.assertEqual(len(results), 3)
        self.assertIn("M-137 (Michigan highway)", [r["title"] for r in results])

    def test_marqo_documents(self):
        try:
            index = MarqoIndexer(self.MARQO_SERVER, self.TEST_INDEX_NAME)
        except Indexer.IndexNotFoundError:
            print(f"Index {self.TEST_INDEX_NAME} not found, repopulate")
            self.test_marqo_populate()
            index = MarqoIndexer(self.MARQO_SERVER, self.TEST_INDEX_NAME)

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(index.search("What is M-137?"))
        self.showMarqoResults(results)
        self.assertIn("7751000", [r["_id"] for r in results])

        m137 = loop.run_until_complete(index.get(["7751000", "not there"]))
        self.assertEqual([True, False], [r["_found"] for r in m137])

        deleteStatus = loop.run_until_complete(index.delete("7751062"))
        self.assertEqual(len(deleteStatus["items"]), 1)

        deleted = loop.run_until_complete(index.get("7751062"))
        self.assertEqual([False], [r["_found"] for r in deleted])

        results = loop.run_until_complete(index.search("Attraction in Michigan"))
        self.showMarqoResults(results)
        self.assertNotIn("7751062", [r["_id"] for r in results])


if __name__ == '__main__':
    unittest.main()
