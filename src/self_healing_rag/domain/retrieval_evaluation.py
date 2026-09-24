from collections.abc import Sequence
from dataclasses import dataclass

from self_healing_rag.domain.failures import FailureType


@dataclass(frozen=True)
class RetrievalEvaluation:
    success: bool
    failure_type: FailureType | None
    reason: str
    document_count: int
    top_score: float | None


def evaluate_retrieval(
    scores: Sequence[float],
    min_relevance_score: float,
) -> RetrievalEvaluation:
    if not -1.0 <= min_relevance_score <= 1.0:
        raise ValueError("min_relevance_score must be between -1 and 1")

    document_count = len(scores)

    if document_count == 0:
        return RetrievalEvaluation(
            success=False,
            failure_type=FailureType.NO_DOCUMENTS,
            reason="No documents were retrieved.",
            document_count=0,
            top_score=None,
        )

    top_score = max(scores)

    if top_score < min_relevance_score:
        return RetrievalEvaluation(
            success=False,
            failure_type=FailureType.LOW_RELEVANCE,
            reason=(
                f"Top retrieval score {top_score:.4f} is below "
                f"the minimum relevance score {min_relevance_score:.4f}."
            ),
            document_count=document_count,
            top_score=top_score,
        )

    return RetrievalEvaluation(
        success=True,
        failure_type=None,
        reason="Retrieved documents meet the minimum relevance threshold.",
        document_count=document_count,
        top_score=top_score,
    )
