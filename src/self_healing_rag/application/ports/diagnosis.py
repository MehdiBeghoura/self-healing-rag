from dataclasses import dataclass
from typing import Protocol

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.failures import FailureType


@dataclass(frozen=True)
class DiagnosisRequest:
    query: str
    failure_type: FailureType
    reason: str
    retrieved_chunks: list[RetrievedChunk]


class DiagnosticProvider(Protocol):
    def diagnose(self, request: DiagnosisRequest) -> Diagnosis: ...
