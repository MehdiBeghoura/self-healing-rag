import json
from types import SimpleNamespace

from self_healing_rag.application.ports.diagnosis import DiagnosisRequest
from self_healing_rag.domain.failures import FailureType
from self_healing_rag.domain.recovery import RecoveryAction
from self_healing_rag.infrastructure.llm import diagnosis as diagnosis_module


class FakeOllamaClient:
    def __init__(self, response_content: str) -> None:
        self.response_content = response_content

    def chat(self, **kwargs):
        return SimpleNamespace(message=SimpleNamespace(content=self.response_content))


def make_request() -> DiagnosisRequest:
    return DiagnosisRequest(
        original_query="What is the capital city of France?",
        retrieval_query="What is the capital city of France?",
        failure_type=FailureType.LOW_RELEVANCE,
        reason="The top retrieval score is below the threshold.",
        retrieved_chunks=[],
    )


def test_stop_discards_rewritten_query(monkeypatch) -> None:
    fake_client = FakeOllamaClient(
        json.dumps(
            {
                "explanation": "The question is outside the document collection.",
                "recommended_action": "STOP",
                "rewritten_query": "France capital city",
            }
        )
    )

    monkeypatch.setattr(
        diagnosis_module,
        "Client",
        lambda host: fake_client,
    )

    client = diagnosis_module.OllamaDiagnosticClient()
    diagnosis = client.diagnose(make_request())

    assert diagnosis.recommended_action == RecoveryAction.STOP
    assert diagnosis.rewritten_query is None


def test_rewrite_action_preserves_rewritten_query(monkeypatch) -> None:
    rewritten_query = "PostgreSQL employee salary data"

    fake_client = FakeOllamaClient(
        json.dumps(
            {
                "explanation": "A more specific query may improve retrieval.",
                "recommended_action": "REWRITE_QUERY",
                "rewritten_query": rewritten_query,
            }
        )
    )

    monkeypatch.setattr(
        diagnosis_module,
        "Client",
        lambda host: fake_client,
    )

    client = diagnosis_module.OllamaDiagnosticClient()
    diagnosis = client.diagnose(make_request())

    assert diagnosis.recommended_action == RecoveryAction.REWRITE_QUERY
    assert diagnosis.rewritten_query == rewritten_query
