from uuid import uuid4

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalRequest,
    EvaluateRetrievalUseCase,
)
from self_healing_rag.domain.failures import FailureType


def make_chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="Test content",
        source="test.md",
        title="Test Document",
        score=score,
        metadata={},
    )


def test_evaluate_retrieval_passes_scores_to_domain_policy():
    use_case = EvaluateRetrievalUseCase()

    result = use_case.execute(
        EvaluateRetrievalRequest(
            chunks=[
                make_chunk(0.85),
                make_chunk(0.72),
            ],
            min_relevance_score=0.6,
        )
    )

    assert result.success is True
    assert result.failure_type is None
    assert result.document_count == 2
    assert result.top_score == 0.85


def test_evaluate_retrieval_reports_no_documents():
    use_case = EvaluateRetrievalUseCase()

    result = use_case.execute(
        EvaluateRetrievalRequest(
            chunks=[],
            min_relevance_score=0.6,
        )
    )

    assert result.success is False
    assert result.failure_type == FailureType.NO_DOCUMENTS
    assert result.document_count == 0
