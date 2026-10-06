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
from self_healing_rag.infrastructure.llm.diagnosis import (
    OllamaDiagnosticClient,
)
from self_healing_rag.infrastructure.llm.embeddings import OllamaEmbeddingClient
from self_healing_rag.infrastructure.llm.generation import OllamaGenerationClient
from self_healing_rag.infrastructure.retrieval.pgvector_retriever import (
    PgVectorRetriever,
)
from self_healing_rag.orchestration.rag_graph import build_rag_graph


def main() -> None:
    embedding_client = OllamaEmbeddingClient()
    diagnostic_client = OllamaDiagnosticClient()
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

    rag_graph = build_rag_graph(
        retrieval_use_case=retrieve_use_case,
        evaluation_use_case=EvaluateRetrievalUseCase(),
        diagnostic_provider=diagnostic_client,
        generation_provider=generation_client,
        max_retries=2,
    )

    document_id = ingest_use_case.execute(
        IngestDocumentRequest(
            title="PostgreSQL Vector Guide",
            source="self-healing-smoke-test",
            content=(
                "PostgreSQL can use the pgvector extension to store vector "
                "embeddings and perform similarity search over those "
                "embeddings."
            ),
        )
    )

    try:
        result = rag_graph.invoke(
            {
                "query": "How can PostgreSQL store employee salaries?",
                "top_k": 3,
                "min_relevance_score": 0.6,
            }
        )

        print("\n=== Self-Healing RAG Result ===")
        print("Final query:", result["query"])
        print("Retry count:", result["retry_count"])
        print("Evaluation:", result["evaluation"])
        print("Diagnosis:", result.get("diagnosis"))
        print("Recovery decision:", result.get("recovery_decision"))
        print("Answer:", result["answer"])

    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Document).where(Document.id == document_id))


if __name__ == "__main__":
    main()