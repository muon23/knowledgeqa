import re
from dataclasses import dataclass
from typing import List


@dataclass
class Answer:
    """Answer from a Bot when asked it a question.

    Attributes:
        content (str): The text of the answer
        citations (List[str]): If facts are given to the Bot, this is a list of fact IDs being referenced.
    """
    content: str
    citations: List[str]

    @classmethod
    def of(cls, text: str) -> "Answer":
        """Creates an Answer object from a text containing citations."""
        pattern = r'\[([^\]]+)\]'
        citations = re.findall(pattern, text)

        cleaned = re.sub(pattern, '', text) if citations else text
        return Answer(cleaned.strip(), citations)

    def __str__(self) -> str:
        """Convert to a pretty string"""
        return f"{self.content} [{','.join(self.citations)}]" if self.citations else self.content

