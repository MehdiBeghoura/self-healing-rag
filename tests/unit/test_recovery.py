from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.failures import FailureType
from self_healing_rag.domain.recovery import (
    RecoveryAction,
    decide_recovery,
)


def make_diagnosis(
    failure_type: FailureType,
    action: RecoveryAction,
    rewritten_query: str | None,
) -> Diagnosis:
    return Diagnosis(
        failure_type=failure_type,
        explanation="Test explanation",
        recommended_action=action,
        rewritten_query=rewritten_query,
    )


def test_low_relevance_can_rewrite_query():
    diagnosis = make_diagnosis(
        failure_type=FailureType.LOW_RELEVANCE,
        action=RecoveryAction.REWRITE_QUERY,
        rewritten_query="How does PostgreSQL store employee data?",
    )

    result = decide_recovery(
        original_query="How can PostgreSQL store employee salaries?",
        diagnosis=diagnosis,
        retry_count=0,
        max_retries=2,
    )

    assert result.action == RecoveryAction.REWRITE_QUERY
    assert result.rewritten_query == ("How does PostgreSQL store employee data?")


def test_no_documents_can_rewrite_query():
    diagnosis = make_diagnosis(
        failure_type=FailureType.NO_DOCUMENTS,
        action=RecoveryAction.REWRITE_QUERY,
        rewritten_query="PostgreSQL employee salary storage",
    )

    result = decide_recovery(
        original_query="How can PostgreSQL store employee salaries?",
        diagnosis=diagnosis,
        retry_count=0,
        max_retries=2,
    )

    assert result.action == RecoveryAction.REWRITE_QUERY
    assert result.rewritten_query == ("PostgreSQL employee salary storage")


def test_policy_stops_when_retry_limit_is_reached():
    diagnosis = make_diagnosis(
        failure_type=FailureType.LOW_RELEVANCE,
        action=RecoveryAction.REWRITE_QUERY,
        rewritten_query="A better query",
    )

    result = decide_recovery(
        original_query="Original query",
        diagnosis=diagnosis,
        retry_count=2,
        max_retries=2,
    )

    assert result.action == RecoveryAction.STOP
    assert result.rewritten_query is None


def test_policy_stops_when_action_is_stop():
    diagnosis = make_diagnosis(
        failure_type=FailureType.LOW_RELEVANCE,
        action=RecoveryAction.STOP,
        rewritten_query=None,
    )

    result = decide_recovery(
        original_query="Original query",
        diagnosis=diagnosis,
        retry_count=0,
        max_retries=2,
    )

    assert result.action == RecoveryAction.STOP
    assert result.rewritten_query is None


def test_policy_stops_when_rewritten_query_is_missing():
    diagnosis = make_diagnosis(
        failure_type=FailureType.LOW_RELEVANCE,
        action=RecoveryAction.REWRITE_QUERY,
        rewritten_query=None,
    )

    result = decide_recovery(
        original_query="Original query",
        diagnosis=diagnosis,
        retry_count=0,
        max_retries=2,
    )

    assert result.action == RecoveryAction.STOP
    assert result.rewritten_query is None


def test_policy_stops_when_rewritten_query_is_unchanged():
    diagnosis = make_diagnosis(
        failure_type=FailureType.LOW_RELEVANCE,
        action=RecoveryAction.REWRITE_QUERY,
        rewritten_query="Original query",
    )

    result = decide_recovery(
        original_query="Original query",
        diagnosis=diagnosis,
        retry_count=0,
        max_retries=2,
    )

    assert result.action == RecoveryAction.STOP
    assert result.rewritten_query is None
