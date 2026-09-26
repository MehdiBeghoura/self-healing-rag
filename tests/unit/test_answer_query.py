from uuid import uuid4

import pytest

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.answer_query import (
    AnswerQueryRequest,
    AnswerQueryUseCase,
)
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalUseCase,
)


class FakeRetrievalUseCase:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.last_request = None

    def execute(self, request):
        self.last_request = request
        return self.chunks


class FakeGenerationProvider:
    def __init__(self) -> None:
        self.last_query = None
        self.last_context = None
        self.call_count = 0

    def generate(self, query: str, context: str) -> str:
        self.call_count += 1
        self.last_query = query
        self.last_context = context
        return "Generated answer"


def make_chunk(score: float, content: str = "Test content") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        source="test.md",
        title="Test Document",
        score=score,
        metadata={},
    )


def make_use_case(
    chunks: list[RetrievedChunk],
) -> tuple[AnswerQueryUseCase, FakeRetrievalUseCase, FakeGenerationProvider]:
    retrieval = FakeRetrievalUseCase(chunks)
    generation = FakeGenerationProvider()
    evaluation = EvaluateRetrievalUseCase()

    use_case = AnswerQueryUseCase(
        retrieval_use_case=retrieval,
        evaluation_use_case=evaluation,
        generation_provider=generation,
    )

    return use_case, retrieval, generation


def test_answer_query_retrieves_context_and_generates_answer():
    use_case, retrieval, generation = make_use_case(
        [
            make_chunk(
                0.9,
                "PostgreSQL can store vectors using pgvector.",
            )
        ]
    )

    result = use_case.execute(
        AnswerQueryRequest(
            query="How does PostgreSQL store vectors?",
            top_k=3,
            min_relevance_score=0.6,
        )
    )

    assert result.answer == "Generated answer"
    assert result.evaluation.success is True
    assert result.evaluation.failure_type is None
    assert len(result.retrieved_chunks) == 1

    assert retrieval.last_request.query == "How does PostgreSQL store vectors?"
    assert retrieval.last_request.top_k == 3

    assert generation.call_count == 1
    assert generation.last_query == "How does PostgreSQL store vectors?"
    assert "PostgreSQL can store vectors using pgvector." in generation.last_context


def test_answer_query_returns_failure_without_generating():
    use_case, _, generation = make_use_case(
        [
            make_chunk(0.3),
        ]
    )

    result = use_case.execute(
        AnswerQueryRequest(
            query="What is unrelated?",
            min_relevance_score=0.6,
        )
    )

    assert result.answer is None
    assert result.evaluation.success is False
    assert result.evaluation.failure_type.value == "LOW_RELEVANCE"
    assert generation.call_count == 0


def test_answer_query_returns_no_documents_without_generating():
    use_case, _, generation = make_use_case([])

    result = use_case.execute(
        AnswerQueryRequest(
            query="What is PostgreSQL?",
            min_relevance_score=0.6,
        )
    )

    assert result.answer is None
    assert result.evaluation.success is False
    assert result.evaluation.failure_type.value == "NO_DOCUMENTS"
    assert generation.call_count == 0


@pytest.mark.parametrize(
    "query",
    ["", "   "],
)
def test_answer_query_rejects_empty_query(query: str):
    use_case, _, _ = make_use_case([])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        use_case.execute(AnswerQueryRequest(query=query))
