from dataclasses import dataclass

from self_healing_rag.application.ports.ingestion import EmbeddingProvider
from self_healing_rag.application.ports.retrieval import (
    RetrievedChunk,
    Retriever,
)


@dataclass(frozen=True)
class RetrieveDocumentsRequest:
    query: str
    top_k: int = 5


class RetrieveDocumentsUseCase:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        retriever: Retriever,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._retriever = retriever

    def execute(
        self,
        request: RetrieveDocumentsRequest,
    ) -> list[RetrievedChunk]:
        query = request.query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        if request.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self._embedding_provider.embed_query(query)

        return self._retriever.retrieve(
            embedding=query_embedding,
            top_k=request.top_k,
        )
