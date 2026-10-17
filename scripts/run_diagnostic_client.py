from uuid import uuid4

from self_healing_rag.application.ports.diagnosis import DiagnosisRequest
from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.domain.failures import FailureType
from self_healing_rag.infrastructure.llm.diagnosis import (
    OllamaDiagnosticClient,
)

chunk = RetrievedChunk(
    chunk_id=uuid4(),
    document_id=uuid4(),
    content=(
        "PostgreSQL can use pgvector to store embeddings "
        "and perform similarity search."
    ),
    source="test.md",
    title="PostgreSQL Vector Guide",
    score=0.31,
    metadata={},
)

original_query = "How can PostgreSQL store employee salaries?"

request = DiagnosisRequest(
    original_query=original_query,
    retrieval_query=original_query,
    failure_type=FailureType.LOW_RELEVANCE,
    reason=(
        "Top retrieval score 0.3100 is below the "
        "minimum relevance score 0.4000."
    ),
    retrieved_chunks=[chunk],
)

diagnosis = OllamaDiagnosticClient().diagnose(request)

print("Original query:", request.original_query)
print("Retrieval query:", request.retrieval_query)
print("Failure type:", diagnosis.failure_type)
print("Explanation:", diagnosis.explanation)
print("Recommended action:", diagnosis.recommended_action)
print("Rewritten query:", diagnosis.rewritten_query)