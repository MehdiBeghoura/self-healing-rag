from dataclasses import dataclass

from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.recovery import (
    RecoveryDecision,
    decide_recovery,
)


@dataclass(frozen=True)
class DecideRecoveryRequest:
    original_query: str
    diagnosis: Diagnosis
    retry_count: int
    max_retries: int


class DecideRecoveryUseCase:
    def execute(
        self,
        request: DecideRecoveryRequest,
    ) -> RecoveryDecision:
        return decide_recovery(
            original_query=request.original_query,
            diagnosis=request.diagnosis,
            retry_count=request.retry_count,
            max_retries=request.max_retries,
        )
