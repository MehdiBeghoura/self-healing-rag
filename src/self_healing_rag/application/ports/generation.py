from typing import Protocol


class GenerationProvider(Protocol):
    def generate(self, question: str, context: str) -> str: ...
