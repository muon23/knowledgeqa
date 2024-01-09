import re
from dataclasses import dataclass
from typing import List


@dataclass
class Answer:
    content: str
    citations: List[str]

    @classmethod
    def of(cls, text: str) -> "Answer":
        pattern = r'\[([^\]]+)\]'
        citations = re.findall(pattern, text)

        cleaned = re.sub(pattern, '', text) if citations else text
        return Answer(cleaned.strip(), citations)

    def __str__(self) -> str:
        return f"{self.content} [{','.join(self.citations)}]" if self.citations else self.content

