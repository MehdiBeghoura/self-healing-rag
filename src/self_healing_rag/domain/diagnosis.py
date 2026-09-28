from dataclasses import dataclass

from self_healing_rag.domain.failures import FailureType
from self_healing_rag.domain.recovery import RecoveryAction


@dataclass(frozen=True)
class Diagnosis:
    failure_type: FailureType
    explanation: str
    recommended_action: RecoveryAction
    rewritten_query: str | None