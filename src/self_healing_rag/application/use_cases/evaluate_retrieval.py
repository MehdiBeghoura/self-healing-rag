from dataclasses import dataclass

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.domain.retrieval_evaluation import (
    RetrievalEvaluation,
    evaluate_retrieval,
)


@dataclass(frozen=True)
class EvaluateRetrievalRequest:
    chunks: list[RetrievedChunk]
    min_relevance_score: float


class EvaluateRetrievalUseCase:
    def execute(
        self,
        request: EvaluateRetrievalRequest,
    ) -> RetrievalEvaluation:
        scores = [chunk.score for chunk in request.chunks]

        return evaluate_retrieval(
            scores=scores,
            min_relevance_score=request.min_relevance_score,
        )
