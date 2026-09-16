from sqlalchemy import select

from self_healing_rag.application.use_cases.ingest_document import (
    IngestDocumentRequest,
    IngestDocumentUseCase,
)
from self_healing_rag.infrastructure.database.ingestion import save_document
from self_healing_rag.infrastructure.database.models import Chunk, Document
from self_healing_rag.infrastructure.database.session import SessionLocal
from self_healing_rag.infrastructure.llm.embeddings import OllamaEmbeddingClient


def test_ingest_document_end_to_end() -> None:
    use_case = IngestDocumentUseCase(
        embedding_provider=OllamaEmbeddingClient(),
        document_store=save_document,
    )

    document_id = use_case.execute(
        IngestDocumentRequest(
            title="PostgreSQL Test Guide",
            source="integration-test-real",
            content=(
                "PostgreSQL is a relational database system. "
                "The pgvector extension allows PostgreSQL to store "
                "and search vector embeddings. "
                "Vector similarity can be used for semantic retrieval."
            ),
        )
    )

    try:
        with SessionLocal() as session:
            document = session.get(Document, document_id)

            chunks = session.scalars(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
            ).all()

        assert document is not None
        assert document.title == "PostgreSQL Test Guide"
        assert document.source == "integration-test-real"

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content
            assert len(chunk.embedding) == 1024
            assert chunk.document_id == document_id

    finally:
        with SessionLocal.begin() as session:
            document = session.get(Document, document_id)

            if document is not None:
                session.delete(document)
