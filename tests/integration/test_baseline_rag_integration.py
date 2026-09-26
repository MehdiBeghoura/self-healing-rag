from sqlalchemy import delete

from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalUseCase,
)
from self_healing_rag.application.use_cases.ingest_document import (
    IngestDocumentRequest,
    IngestDocumentUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsUseCase,
)
from self_healing_rag.infrastructure.database.ingestion import save_document
from self_healing_rag.infrastructure.database.models import Document
from self_healing_rag.infrastructure.database.session import SessionLocal
from self_healing_rag.infrastructure.llm.embeddings import OllamaEmbeddingClient
from self_healing_rag.infrastructure.llm.generation import OllamaGenerationClient
from self_healing_rag.infrastructure.retrieval.pgvector_retriever import (
    PgVectorRetriever,
)
from self_healing_rag.orchestration.rag_graph import build_rag_graph


def test_baseline_rag_end_to_end():
    embedding_client = OllamaEmbeddingClient()
    generation_client = OllamaGenerationClient()
    retriever = PgVectorRetriever()

    ingest_use_case = IngestDocumentUseCase(
        embedding_provider=embedding_client,
        document_store=save_document,
    )

    retrieve_use_case = RetrieveDocumentsUseCase(
        embedding_provider=embedding_client,
        retriever=retriever,
    )

    evaluate_use_case = EvaluateRetrievalUseCase()

    rag_graph = build_rag_graph(
        retrieval_use_case=retrieve_use_case,
        evaluation_use_case=evaluate_use_case,
        generation_provider=generation_client,
    )

    document_id = ingest_use_case.execute(
        IngestDocumentRequest(
            title="PostgreSQL Vector Guide",
            source="integration-test",
            content=(
                "PostgreSQL can use the pgvector extension to store vector "
                "embeddings. pgvector also allows applications to perform "
                "similarity search over those embeddings."
            ),
        )
    )

    try:
        result = rag_graph.invoke(
            {
                "query": "What does pgvector allow PostgreSQL to do?",
                "top_k": 3,
                "min_relevance_score": 0.0,
            }
        )

        assert result["retrieved_chunks"]
        assert result["retrieved_chunks"][0]["title"] == ("PostgreSQL Vector Guide")
        assert "pgvector" in result["retrieved_chunks"][0]["content"].lower()

        assert result["evaluation"]["success"] is True
        assert result["evaluation"]["failure_type"] is None

        assert result["answer"] is not None
        assert result["answer"].strip()
        assert "pgvector" in result["answer"].lower()

    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Document).where(Document.id == document_id))
