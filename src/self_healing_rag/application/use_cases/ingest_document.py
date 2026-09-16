from dataclasses import dataclass
from uuid import UUID

from self_healing_rag.application.ports.ingestion import (
    DocumentStore,
    EmbeddingProvider,
)
from self_healing_rag.application.services.chunking import split_text


@dataclass(frozen=True)
class IngestDocumentRequest:
    title: str
    source: str
    content: str


class IngestDocumentUseCase:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        document_store: DocumentStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._document_store = document_store

    def execute(self, request: IngestDocumentRequest) -> UUID:
        chunks = split_text(request.content)

        if not chunks:
            raise ValueError("Document content cannot be empty")

        embeddings = self._embedding_provider.embed_documents(
            [chunk.content for chunk in chunks]
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding provider returned a different number "
                "of embeddings than chunks"
            )

        embedded_chunks = [
            (chunk.index, chunk.content, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        return self._document_store(
            title=request.title,
            source=request.source,
            chunks=embedded_chunks,
        )
