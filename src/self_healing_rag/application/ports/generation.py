from typing import Protocol


class GenerationProvider(Protocol):
    def generate(self, query: str, context: str) -> str: ...
