from uuid import uuid4

import pytest

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.answer_query import (
    AnswerQueryRequest,
    AnswerQueryUseCase,
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
        self.last_question = None
        self.last_context = None

    def generate(self, question: str, context: str) -> str:
        self.last_question = question
        self.last_context = context
        return "Generated answer"


def make_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        source="test.md",
        title="Test Document",
        score=0.9,
        metadata={},
    )


def test_answer_query_retrieves_context_and_generates_answer():
    retrieval = FakeRetrievalUseCase(
        [make_chunk("PostgreSQL can store vectors using pgvector.")]
    )
    generation = FakeGenerationProvider()

    use_case = AnswerQueryUseCase(retrieval, generation)

    result = use_case.execute(
        AnswerQueryRequest(
            query="How does PostgreSQL store vectors?",
            top_k=3,
        )
    )

    assert result == "Generated answer"
    assert retrieval.last_request.query == "How does PostgreSQL store vectors?"
    assert retrieval.last_request.top_k == 3
    assert generation.last_question == "How does PostgreSQL store vectors?"
    assert "PostgreSQL can store vectors using pgvector." in generation.last_context


@pytest.mark.parametrize(
    "query",
    ["", "   "],
)
def test_answer_query_rejects_empty_query(query: str):
    retrieval = FakeRetrievalUseCase([])
    generation = FakeGenerationProvider()

    use_case = AnswerQueryUseCase(retrieval, generation)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        use_case.execute(AnswerQueryRequest(query=query))
