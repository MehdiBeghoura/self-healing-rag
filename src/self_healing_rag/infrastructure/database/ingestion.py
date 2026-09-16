from uuid import UUID

from self_healing_rag.infrastructure.database.models import Chunk, Document
from self_healing_rag.infrastructure.database.session import SessionLocal


def save_document(
    title: str,
    source: str,
    chunks: list[tuple[int, str, list[float]]],
) -> UUID:
    with SessionLocal.begin() as session:
        document = Document(
            title=title,
            source=source,
        )

        session.add(document)
        session.flush()

        for chunk_index, content, embedding in chunks:
            session.add(
                Chunk(
                    document_id=document.id,
                    content=content,
                    chunk_index=chunk_index,
                    embedding=embedding,
                )
            )

        return document.id
