from sqlalchemy import select

from self_healing_rag.infrastructure.database.ingestion import save_document
from self_healing_rag.infrastructure.database.models import Chunk, Document
from self_healing_rag.infrastructure.database.session import SessionLocal


def test_save_document_persists_document_and_chunks() -> None:
    embedding = [0.1] * 1024

    document_id = save_document(
        title="Test document",
        source="integration-test",
        chunks=[
            (0, "First chunk", embedding),
            (1, "Second chunk", embedding),
        ],
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
        assert document.title == "Test document"
        assert document.source == "integration-test"

        assert len(chunks) == 2
        assert chunks[0].content == "First chunk"
        assert chunks[1].content == "Second chunk"
        assert len(chunks[0].embedding) == 1024

    finally:
        with SessionLocal.begin() as session:
            document = session.get(Document, document_id)
            if document is not None:
                session.delete(document)
