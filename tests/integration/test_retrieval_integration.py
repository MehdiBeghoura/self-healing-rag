from self_healing_rag.application.use_cases.ingest_document import (
    IngestDocumentRequest,
    IngestDocumentUseCase,
)
from self_healing_rag.infrastructure.database.ingestion import save_document
from self_healing_rag.infrastructure.database.models import Document
from self_healing_rag.infrastructure.database.session import SessionLocal
from self_healing_rag.infrastructure.llm.embeddings import OllamaEmbeddingClient
from self_healing_rag.infrastructure.retrieval.pgvector_retriever import (
    PgVectorRetriever,
)


def test_retrieval_returns_most_relevant_chunk_first() -> None:
    embedding_client = OllamaEmbeddingClient()

    ingest = IngestDocumentUseCase(
        embedding_provider=embedding_client,
        document_store=save_document,
    )

    documents = [
        IngestDocumentRequest(
            title="PostgreSQL Guide",
            source="retrieval-test-postgresql",
            content="PostgreSQL can store vector embeddings using the pgvector extension.",
        ),
        IngestDocumentRequest(
            title="Python Guide",
            source="retrieval-test-python",
            content="Python is a programming language used to build applications.",
        ),
        IngestDocumentRequest(
            title="Climate Guide",
            source="retrieval-test-climate",
            content="The Mediterranean climate has mild and wet winters.",
        ),
    ]

    document_ids = []

    try:
        for document in documents:
            document_ids.append(ingest.execute(document))

        query_embedding = embedding_client.embed_query(
            "How can PostgreSQL store vector embeddings?"
        )

        retriever = PgVectorRetriever()
        results = retriever.retrieve(
            embedding=query_embedding,
            top_k=3,
        )

        assert len(results) == 3

        assert results[0].title == "PostgreSQL Guide"
        assert results[0].source == "retrieval-test-postgresql"

        assert results[0].score >= results[1].score
        assert results[1].score >= results[2].score

        assert all(result.content for result in results)

    finally:
        with SessionLocal.begin() as session:
            for document_id in document_ids:
                document = session.get(Document, document_id)

                if document is not None:
                    session.delete(document)
