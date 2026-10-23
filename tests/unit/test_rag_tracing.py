from collections.abc import Iterator
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from self_healing_rag.application.ports.diagnosis import DiagnosisRequest
from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
)
from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.failures import FailureType
from self_healing_rag.domain.recovery import RecoveryAction
from self_healing_rag.orchestration import rag_graph as rag_graph_module


@pytest.fixture
def span_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()

    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Use a dedicated test tracer rather than the application's
    # global provider or console exporter.
    monkeypatch.setattr(
        rag_graph_module,
        "tracer",
        provider.get_tracer("test-rag-tracing"),
    )

    yield exporter

    provider.shutdown()


def make_chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=(
            "PostgreSQL can use pgvector to store embeddings "
            "and perform similarity search."
        ),
        source="test.md",
        title="PostgreSQL Vector Guide",
        score=score,
        metadata={},
    )


class FakeRetrievalUseCase:
    def __init__(
        self,
        responses: list[list[RetrievedChunk]],
    ) -> None:
        self.responses = responses
        self.queries: list[str] = []

    def execute(
        self,
        request: RetrieveDocumentsRequest,
    ) -> list[RetrievedChunk]:
        self.queries.append(request.query)

        response_index = min(
            len(self.queries) - 1,
            len(self.responses) - 1,
        )
        return self.responses[response_index]


class FakeDiagnosticProvider:
    def __init__(self, diagnosis: Diagnosis) -> None:
        self.diagnosis = diagnosis
        self.call_count = 0

    def diagnose(self, request: DiagnosisRequest) -> Diagnosis:
        self.call_count += 1
        return self.diagnosis


class FakeGenerationProvider:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, query: str, context: str) -> str:
        self.call_count += 1
        return f"Answer to: {query}"


def make_diagnosis(
    action: RecoveryAction,
    rewritten_query: str | None = None,
) -> Diagnosis:
    return Diagnosis(
        failure_type=FailureType.LOW_RELEVANCE,
        explanation="The retrieved evidence has low relevance.",
        recommended_action=action,
        rewritten_query=rewritten_query,
    )


def invoke_graph(graph):
    return graph.invoke(
        {
            "query": "What does pgvector do?",
            "top_k": 3,
            "min_relevance_score": 0.4,
        }
    )


def test_successful_retrieval_emits_expected_spans(
    span_exporter: InMemorySpanExporter,
) -> None:
    retrieval = FakeRetrievalUseCase([[make_chunk(0.8)]])
    diagnostic = FakeDiagnosticProvider(make_diagnosis(RecoveryAction.STOP))
    generation = FakeGenerationProvider()

    graph = rag_graph_module.build_rag_graph(
        retrieval_use_case=retrieval,
        evaluation_use_case=EvaluateRetrievalUseCase(),
        diagnostic_provider=diagnostic,
        generation_provider=generation,
        max_retries=2,
    )

    result = invoke_graph(graph)

    spans = span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert {
        "rag.retrieve",
        "rag.evaluate_retrieval",
        "rag.generate_answer",
    }.issubset(spans_by_name)

    assert spans_by_name["rag.retrieve"].attributes["rag.retrieved_chunk_count"] == 1

    assert (
        spans_by_name["rag.evaluate_retrieval"].attributes["rag.retrieval.success"]
        is True
    )

    assert (
        spans_by_name["rag.generate_answer"].attributes["rag.generation.answer_present"]
        is True
    )

    assert result["evaluation"]["success"] is True
    assert result["answer"] is not None
    assert generation.call_count == 1
    assert diagnostic.call_count == 0


def test_failed_retrieval_emits_stop_decision_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    retrieval = FakeRetrievalUseCase([[make_chunk(0.2)]])
    diagnostic = FakeDiagnosticProvider(make_diagnosis(RecoveryAction.STOP))
    generation = FakeGenerationProvider()

    graph = rag_graph_module.build_rag_graph(
        retrieval_use_case=retrieval,
        evaluation_use_case=EvaluateRetrievalUseCase(),
        diagnostic_provider=diagnostic,
        generation_provider=generation,
        max_retries=2,
    )

    result = invoke_graph(graph)

    spans = span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert {
        "rag.retrieve",
        "rag.evaluate_retrieval",
        "rag.diagnose",
        "rag.decide_recovery",
    }.issubset(spans_by_name)

    assert (
        spans_by_name["rag.evaluate_retrieval"].attributes["rag.retrieval.failure_type"]
        == "LOW_RELEVANCE"
    )

    assert (
        spans_by_name["rag.diagnose"].attributes["rag.diagnosis.recommended_action"]
        == "STOP"
    )

    assert (
        spans_by_name["rag.decide_recovery"].attributes["rag.recovery.action"] == "STOP"
    )

    assert "rag.apply_recovery" not in spans_by_name
    assert result["answer"] is None
    assert result["retry_count"] == 0
    assert generation.call_count == 0


def test_rewritten_query_emits_retry_spans(
    span_exporter: InMemorySpanExporter,
) -> None:
    original_query = "What does pgvector do?"
    rewritten_query = (
        "How does pgvector store embeddings and perform similarity search?"
    )

    retrieval = FakeRetrievalUseCase(
        [
            [make_chunk(0.2)],
            [make_chunk(0.8)],
        ]
    )
    diagnostic = FakeDiagnosticProvider(
        make_diagnosis(
            RecoveryAction.REWRITE_QUERY,
            rewritten_query,
        )
    )
    generation = FakeGenerationProvider()

    graph = rag_graph_module.build_rag_graph(
        retrieval_use_case=retrieval,
        evaluation_use_case=EvaluateRetrievalUseCase(),
        diagnostic_provider=diagnostic,
        generation_provider=generation,
        max_retries=2,
    )

    result = invoke_graph(graph)

    spans = span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    retrieval_spans = [span for span in spans if span.name == "rag.retrieve"]

    assert len(retrieval_spans) == 2
    assert [span.attributes["rag.retry_count"] for span in retrieval_spans] == [0, 1]

    assert {
        "rag.diagnose",
        "rag.decide_recovery",
        "rag.apply_recovery",
        "rag.generate_answer",
    }.issubset(spans_by_name)

    assert (
        spans_by_name["rag.decide_recovery"].attributes["rag.recovery.action"]
        == "REWRITE_QUERY"
    )

    assert (
        spans_by_name["rag.apply_recovery"].attributes["rag.recovery.rewrite_applied"]
        is True
    )

    assert retrieval.queries == [
        original_query,
        rewritten_query,
    ]
    assert result["query"] == original_query
    assert result["retrieval_query"] == rewritten_query
    assert result["retry_count"] == 1
    assert result["evaluation"]["success"] is True
    assert generation.call_count == 1
