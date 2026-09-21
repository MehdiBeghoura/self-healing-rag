from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    source: str
    title: str
    score: float
    metadata: dict


class Retriever(Protocol):
    def retrieve(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]: ...
