from typing import TypedDict
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from self_healing_rag.application.ports.run_inspection import (
    RunInspection,
)
from self_healing_rag.infrastructure.database.run_inspection import (
    CheckpointRunInspectionReader,
)


class InspectionState(TypedDict):
    evaluation: dict[str, object]
    retrieved_chunks: list[dict[str, object]]
    min_relevance_score: float
    retry_count: int
    diagnosis: dict[str, object] | None
    recovery_decision: dict[str, object] | None
    answer: str | None


def _complete_run(_state: InspectionState) -> dict:
    return {}


def save_checkpoint(
    checkpointer: InMemorySaver,
    run_id: UUID,
    state: InspectionState,
) -> None:
    builder = StateGraph(InspectionState)

    builder.add_node("complete", _complete_run)
    builder.add_edge(START, "complete")
    builder.add_edge("complete", END)

    graph = builder.compile(checkpointer=checkpointer)

    graph.invoke(
        state,
        config={
            "configurable": {
                "thread_id": str(run_id),
            }
        },
    )


def test_reader_returns_successful_run_inspection() -> None:
    run_id = uuid4()
    checkpointer = InMemorySaver()

    save_checkpoint(
        checkpointer,
        run_id,
        {
            "evaluation": {
                "success": True,
                "failure_type": None,
                "reason": "Retrieval passed.",
                "document_count": 2,
                "top_score": 0.7294,
            },
            "retrieved_chunks": [
                {"content": "Private document content 1"},
                {"content": "Private document content 2"},
            ],
            "min_relevance_score": 0.4,
            "retry_count": 0,
            "diagnosis": None,
            "recovery_decision": None,
            "answer": "A generated answer.",
        },
    )

    reader = CheckpointRunInspectionReader(checkpointer)

    result = reader.get_run_inspection(run_id)

    assert result == RunInspection(
        run_id=run_id,
        retrieval_success=True,
        failure_type=None,
        retrieved_chunk_count=2,
        top_score=0.7294,
        minimum_relevance_score=0.4,
        recommended_action=None,
        recovery_action=None,
        retry_count=0,
        answer_generated=True,
    )

    # The inspection result must not expose retrieved document content.
    assert not hasattr(result, "retrieved_chunks")


def test_reader_returns_failed_run_with_diagnosis() -> None:
    run_id = uuid4()
    checkpointer = InMemorySaver()

    save_checkpoint(
        checkpointer,
        run_id,
        {
            "evaluation": {
                "success": False,
                "failure_type": "LOW_RELEVANCE",
                "reason": "Top score below threshold.",
                "document_count": 3,
                "top_score": 0.2455,
            },
            "retrieved_chunks": [
                {"content": "Private document content 1"},
                {"content": "Private document content 2"},
                {"content": "Private document content 3"},
            ],
            "min_relevance_score": 0.4,
            "retry_count": 0,
            "diagnosis": {
                "failure_type": "LOW_RELEVANCE",
                "explanation": "Retrieved documents are unrelated.",
                "recommended_action": "STOP",
                "rewritten_query": None,
            },
            "recovery_decision": {
                "action": "STOP",
                "rewritten_query": None,
                "reason": "Diagnostic recommendation does not permit query rewriting.",
            },
            "answer": None,
        },
    )

    reader = CheckpointRunInspectionReader(checkpointer)

    result = reader.get_run_inspection(run_id)

    assert result == RunInspection(
        run_id=run_id,
        retrieval_success=False,
        failure_type="LOW_RELEVANCE",
        retrieved_chunk_count=3,
        top_score=0.2455,
        minimum_relevance_score=0.4,
        recommended_action="STOP",
        recovery_action="STOP",
        retry_count=0,
        answer_generated=False,
    )


def test_reader_returns_none_when_run_does_not_exist() -> None:
    checkpointer = InMemorySaver()
    reader = CheckpointRunInspectionReader(checkpointer)

    result = reader.get_run_inspection(uuid4())

    assert result is None
