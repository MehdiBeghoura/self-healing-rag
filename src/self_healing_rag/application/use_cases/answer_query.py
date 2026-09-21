from dataclasses import dataclass

from self_healing_rag.application.ports.generation import GenerationProvider
from self_healing_rag.application.services.context_builder import build_context
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
    RetrieveDocumentsUseCase,
)


@dataclass(frozen=True)
class AnswerQueryRequest:
    query: str
    top_k: int = 5


class AnswerQueryUseCase:
    def __init__(
        self,
        retrieval_use_case: RetrieveDocumentsUseCase,
        generation_provider: GenerationProvider,
    ) -> None:
        self._retrieval_use_case = retrieval_use_case
        self._generation_provider = generation_provider

    def execute(self, request: AnswerQueryRequest) -> str:
        query = request.query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        chunks = self._retrieval_use_case.execute(
            RetrieveDocumentsRequest(
                query=query,
                top_k=request.top_k,
            )
        )

        context = build_context(chunks)

        return self._generation_provider.generate(
            question=query,
            context=context,
        )
