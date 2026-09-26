from uuid import uuid4

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalUseCase,
)
from self_healing_rag.orchestration.rag_graph import build_rag_graph


class FakeRetrievalUseCase:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def execute(self, request):
        return self.chunks


class FakeGenerationProvider:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, query: str, context: str) -> str:
        self.call_count += 1
        return f"Answer to: {query}"


def make_chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="PostgreSQL uses pgvector for vector similarity search.",
        source="test.md",
        title="PostgreSQL Guide",
        score=score,
        metadata={},
    )


def build_test_graph(
    chunks: list[RetrievedChunk],
    generation: FakeGenerationProvider,
):
    retrieval = FakeRetrievalUseCase(chunks)

    return build_rag_graph(
        retrieval_use_case=retrieval,
        evaluation_use_case=EvaluateRetrievalUseCase(),
        generation_provider=generation,
    )


def test_graph_generates_answer_when_retrieval_succeeds():
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        chunks=[make_chunk(0.9)],
        generation=generation,
    )

    result = graph.invoke(
        {
            "query": "What does pgvector do?",
            "top_k": 3,
            "min_relevance_score": 0.6,
        }
    )

    assert result["evaluation"]["success"] is True
    assert result["evaluation"]["failure_type"] is None
    assert result["answer"] == "Answer to: What does pgvector do?"
    assert len(result["retrieved_chunks"]) == 1
    assert generation.call_count == 1


def test_graph_stops_when_retrieval_has_low_relevance():
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        chunks=[make_chunk(0.3)],
        generation=generation,
    )

    result = graph.invoke(
        {
            "query": "What does pgvector do?",
            "top_k": 3,
            "min_relevance_score": 0.6,
        }
    )

    assert result["evaluation"]["success"] is False
    assert result["evaluation"]["failure_type"] == "LOW_RELEVANCE"
    assert result["answer"] is None
    assert generation.call_count == 0


def test_graph_stops_when_no_documents_are_retrieved():
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        chunks=[],
        generation=generation,
    )

    result = graph.invoke(
        {
            "query": "What does pgvector do?",
            "top_k": 3,
            "min_relevance_score": 0.6,
        }
    )

    assert result["evaluation"]["success"] is False
    assert result["evaluation"]["failure_type"] == "NO_DOCUMENTS"
    assert result["answer"] is None
    assert generation.call_count == 0
