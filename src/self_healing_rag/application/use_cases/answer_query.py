from dataclasses import dataclass

from self_healing_rag.application.ports.generation import GenerationProvider
from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.services.context_builder import build_context
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalRequest,
    EvaluateRetrievalUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
    RetrieveDocumentsUseCase,
)
from self_healing_rag.domain.retrieval_evaluation import RetrievalEvaluation


@dataclass(frozen=True)
class AnswerQueryRequest:
    query: str
    top_k: int = 5
    min_relevance_score: float = 0.6


@dataclass(frozen=True)
class AnswerQueryResult:
    answer: str | None
    retrieved_chunks: list[RetrievedChunk]
    evaluation: RetrievalEvaluation


class AnswerQueryUseCase:
    def __init__(
        self,
        retrieval_use_case: RetrieveDocumentsUseCase,
        evaluation_use_case: EvaluateRetrievalUseCase,
        generation_provider: GenerationProvider,
    ) -> None:
        self._retrieval_use_case = retrieval_use_case
        self._evaluation_use_case = evaluation_use_case
        self._generation_provider = generation_provider

    def execute(self, request: AnswerQueryRequest) -> AnswerQueryResult:
        query = request.query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        chunks = self._retrieval_use_case.execute(
            RetrieveDocumentsRequest(
                query=query,
                top_k=request.top_k,
            )
        )

        evaluation = self._evaluation_use_case.execute(
            EvaluateRetrievalRequest(
                chunks=chunks,
                min_relevance_score=request.min_relevance_score,
            )
        )

        if not evaluation.success:
            return AnswerQueryResult(
                answer=None,
                retrieved_chunks=chunks,
                evaluation=evaluation,
            )

        context = build_context(chunks)

        answer = self._generation_provider.generate(
            query=query,
            context=context,
        )

        return AnswerQueryResult(
            answer=answer,
            retrieved_chunks=chunks,
            evaluation=evaluation,
        )
