from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class RunInspection:
    """Safe diagnostic summary of a completed RAG execution."""

    run_id: UUID

    retrieval_success: bool
    failure_type: str | None

    retrieved_chunk_count: int
    top_score: float | None
    minimum_relevance_score: float

    recommended_action: str | None
    recovery_action: str | None

    retry_count: int
    answer_generated: bool


class RunInspectionReader(Protocol):
    """Port for retrieving diagnostic information about a RAG run."""

    def get_run_inspection(
        self,
        run_id: UUID,
    ) -> RunInspection | None:
        """Return a run inspection, or None if the run does not exist."""
        ...