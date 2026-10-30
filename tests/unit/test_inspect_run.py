from uuid import UUID, uuid4

from self_healing_rag.application.ports.run_inspection import (
    RunInspection,
)
from self_healing_rag.application.use_cases.inspect_run import (
    InspectRunRequest,
    InspectRunUseCase,
)


class FakeRunInspectionReader:
    def __init__(
        self,
        inspection: RunInspection | None,
    ) -> None:
        self.inspection = inspection
        self.received_run_id: UUID | None = None

    def get_run_inspection(
        self,
        run_id: UUID,
    ) -> RunInspection | None:
        self.received_run_id = run_id
        return self.inspection


def test_inspect_run_returns_existing_inspection() -> None:
    run_id = uuid4()

    expected_inspection = RunInspection(
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

    reader = FakeRunInspectionReader(expected_inspection)
    use_case = InspectRunUseCase(reader)

    result = use_case.execute(InspectRunRequest(run_id=run_id))

    assert result == expected_inspection
    assert reader.received_run_id == run_id


def test_inspect_run_returns_none_when_run_does_not_exist() -> None:
    run_id = uuid4()

    reader = FakeRunInspectionReader(None)
    use_case = InspectRunUseCase(reader)

    result = use_case.execute(InspectRunRequest(run_id=run_id))

    assert result is None
    assert reader.received_run_id == run_id
