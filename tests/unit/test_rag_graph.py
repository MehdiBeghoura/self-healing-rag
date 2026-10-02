from uuid import uuid4

from self_healing_rag.application.ports.diagnosis import DiagnosisRequest
from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalUseCase,
)
from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.recovery import RecoveryAction
from self_healing_rag.orchestration.rag_graph import build_rag_graph


class FakeRetrievalUseCase:
    def __init__(self, responses: list[list[RetrievedChunk]]) -> None:
        self.responses = responses
        self.call_count = 0
        self.queries: list[str] = []

    def execute(self, request):
        self.queries.append(request.query)

        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1

        return response


class FakeDiagnosticProvider:
    def __init__(
        self,
        actions: list[tuple[RecoveryAction, str | None]],
    ) -> None:
        self.actions = actions
        self.call_count = 0

    def diagnose(self, request: DiagnosisRequest) -> Diagnosis:
        action, rewritten_query = self.actions[
            min(self.call_count, len(self.actions) - 1)
        ]
        self.call_count += 1

        return Diagnosis(
            failure_type=request.failure_type,
            explanation="Test diagnosis",
            recommended_action=action,
            rewritten_query=rewritten_query,
        )


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
    retrieval: FakeRetrievalUseCase,
    diagnostic: FakeDiagnosticProvider,
    generation: FakeGenerationProvider,
    max_retries: int = 2,
):
    return build_rag_graph(
        retrieval_use_case=retrieval,
        evaluation_use_case=EvaluateRetrievalUseCase(),
        diagnostic_provider=diagnostic,
        generation_provider=generation,
        max_retries=max_retries,
    )


def test_graph_generates_answer_when_retrieval_succeeds():
    retrieval = FakeRetrievalUseCase([[make_chunk(0.9)]])
    diagnostic = FakeDiagnosticProvider([])
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        retrieval=retrieval,
        diagnostic=diagnostic,
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
    assert result["answer"] == "Answer to: What does pgvector do?"
    assert result["retry_count"] == 0
    assert diagnostic.call_count == 0
    assert generation.call_count == 1


def test_graph_rewrites_query_and_retries():
    retrieval = FakeRetrievalUseCase(
        [
            [make_chunk(0.3)],
            [make_chunk(0.9)],
        ]
    )
    diagnostic = FakeDiagnosticProvider(
        [
            (
                RecoveryAction.REWRITE_QUERY,
                "How does PostgreSQL store employee data?",
            )
        ]
    )
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        retrieval=retrieval,
        diagnostic=diagnostic,
        generation=generation,
    )

    result = graph.invoke(
        {
            "query": "How can PostgreSQL store employee salaries?",
            "top_k": 3,
            "min_relevance_score": 0.6,
        }
    )

    assert result["evaluation"]["success"] is True
    assert result["answer"] is not None
    assert result["retry_count"] == 1
    assert retrieval.call_count == 2
    assert retrieval.queries == [
        "How can PostgreSQL store employee salaries?",
        "How does PostgreSQL store employee data?",
    ]
    assert diagnostic.call_count == 1
    assert generation.call_count == 1


def test_graph_stops_when_diagnosis_recommends_stop():
    retrieval = FakeRetrievalUseCase([[make_chunk(0.3)]])
    diagnostic = FakeDiagnosticProvider([(RecoveryAction.STOP, None)])
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        retrieval=retrieval,
        diagnostic=diagnostic,
        generation=generation,
    )

    result = graph.invoke(
        {
            "query": "Unrelated question",
            "top_k": 3,
            "min_relevance_score": 0.6,
        }
    )

    assert result["evaluation"]["success"] is False
    assert result["evaluation"]["failure_type"] == "LOW_RELEVANCE"
    assert result["answer"] is None
    assert result["retry_count"] == 0
    assert diagnostic.call_count == 1
    assert generation.call_count == 0


def test_graph_stops_after_max_retries():
    retrieval = FakeRetrievalUseCase(
        [
            [make_chunk(0.3)],
            [make_chunk(0.3)],
            [make_chunk(0.3)],
        ]
    )
    diagnostic = FakeDiagnosticProvider(
        [
            (RecoveryAction.REWRITE_QUERY, "rewritten query 1"),
            (RecoveryAction.REWRITE_QUERY, "rewritten query 2"),
            (RecoveryAction.REWRITE_QUERY, "rewritten query 3"),
        ]
    )
    generation = FakeGenerationProvider()

    graph = build_test_graph(
        retrieval=retrieval,
        diagnostic=diagnostic,
        generation=generation,
        max_retries=2,
    )

    result = graph.invoke(
        {
            "query": "Original query",
            "top_k": 3,
            "min_relevance_score": 0.6,
        }
    )

    assert result["evaluation"]["success"] is False
    assert result["evaluation"]["failure_type"] == "LOW_RELEVANCE"
    assert result["answer"] is None
    assert result["retry_count"] == 2
    assert retrieval.call_count == 3
    assert diagnostic.call_count == 3
    assert generation.call_count == 0
