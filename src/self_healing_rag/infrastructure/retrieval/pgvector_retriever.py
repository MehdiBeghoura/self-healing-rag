from sqlalchemy import select

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.infrastructure.database.models import Chunk, Document
from self_healing_rag.infrastructure.database.session import SessionLocal


class PgVectorRetriever:
    def retrieve(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        distance = Chunk.embedding.cosine_distance(embedding)

        statement = (
            select(Chunk, Document, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .order_by(distance, Chunk.id)
            .limit(top_k)
        )

        with SessionLocal() as session:
            rows = session.execute(statement).all()

        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                source=document.source,
                title=document.title,
                score=1.0 - float(distance_value),
                metadata=chunk.metadata_,
            )
            for chunk, document, distance_value in rows
        ]
