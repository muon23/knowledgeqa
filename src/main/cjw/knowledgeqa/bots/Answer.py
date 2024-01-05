from dataclasses import dataclass


@dataclass
class Answer:
    reference: str
    content: str
