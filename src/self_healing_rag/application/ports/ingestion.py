from typing import Protocol
from uuid import UUID


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class DocumentStore(Protocol):
    def __call__(
        self,
        title: str,
        source: str,
        chunks: list[tuple[int, str, list[float]]],
    ) -> UUID: ...
