from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from self_healing_rag.domain.failures import FailureType

if TYPE_CHECKING:
    from self_healing_rag.domain.diagnosis import Diagnosis


class RecoveryAction(StrEnum):
    REWRITE_QUERY = "REWRITE_QUERY"
    STOP = "STOP"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    rewritten_query: str | None
    reason: str


def decide_recovery(
    original_query: str,
    diagnosis: "Diagnosis",
    retry_count: int,
    max_retries: int,
) -> RecoveryDecision:
    if retry_count < 0:
        raise ValueError("retry_count cannot be negative")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    if retry_count >= max_retries:
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            rewritten_query=None,
            reason="Maximum retry count reached.",
        )

    if diagnosis.failure_type not in {
        FailureType.NO_DOCUMENTS,
        FailureType.LOW_RELEVANCE,
    }:
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            rewritten_query=None,
            reason=(
                f"Recovery is not currently supported for "
                f"{diagnosis.failure_type.value}."
            ),
        )

    if diagnosis.recommended_action != RecoveryAction.REWRITE_QUERY:
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            rewritten_query=None,
            reason="Diagnostic recommendation does not permit query rewriting.",
        )

    rewritten_query = (
        diagnosis.rewritten_query.strip()
        if diagnosis.rewritten_query is not None
        else ""
    )

    if not rewritten_query:
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            rewritten_query=None,
            reason="No usable rewritten query was provided.",
        )

    if rewritten_query == original_query.strip():
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            rewritten_query=None,
            reason="Rewritten query is identical to the original query.",
        )

    return RecoveryDecision(
        action=RecoveryAction.REWRITE_QUERY,
        rewritten_query=rewritten_query,
        reason="Query rewrite is allowed by the recovery policy.",
    )