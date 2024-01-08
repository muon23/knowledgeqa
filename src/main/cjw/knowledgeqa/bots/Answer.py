import re
from dataclasses import dataclass


@dataclass
class Answer:
    content: str
    citation: str | None

    @classmethod
    def of(cls, text: str) -> "Answer":
        pattern = r'\[([^\]]+)\]'
        match = re.search(pattern, text)

        # Check if a match was found
        if match:
            citation = match.group(1)
            cleaned = re.sub(pattern, '', text, count=1)
            return Answer(cleaned.strip(), citation)
        else:
            # Return None if no match was found
            return Answer(text.strip(), None)


