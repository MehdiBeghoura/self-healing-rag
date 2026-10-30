from dataclasses import dataclass
from uuid import UUID

from self_healing_rag.application.ports.run_inspection import (
    RunInspection,
    RunInspectionReader,
)


@dataclass(frozen=True)
class InspectRunRequest:
    run_id: UUID


class InspectRunUseCase:
    def __init__(
        self,
        inspection_reader: RunInspectionReader,
    ) -> None:
        self._inspection_reader = inspection_reader

    def execute(
        self,
        request: InspectRunRequest,
    ) -> RunInspection | None:
        return self._inspection_reader.get_run_inspection(
            run_id=request.run_id,
        )