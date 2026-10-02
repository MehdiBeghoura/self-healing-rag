from typing import NotRequired, TypedDict


class RetrievedChunkState(TypedDict):
    chunk_id: str
    document_id: str
    content: str
    source: str
    title: str
    score: float
    metadata: dict


class RetrievalEvaluationState(TypedDict):
    success: bool
    failure_type: str | None
    reason: str
    document_count: int
    top_score: float | None


class DiagnosisState(TypedDict):
    failure_type: str
    explanation: str
    recommended_action: str
    rewritten_query: str | None


class RecoveryDecisionState(TypedDict):
    action: str
    rewritten_query: str | None
    reason: str


class RAGInputState(TypedDict):
    query: str
    top_k: int
    min_relevance_score: float


class RAGState(RAGInputState):
    retry_count: NotRequired[int]
    retrieved_chunks: NotRequired[list[RetrievedChunkState]]
    evaluation: NotRequired[RetrievalEvaluationState]
    diagnosis: NotRequired[DiagnosisState]
    recovery_decision: NotRequired[RecoveryDecisionState]
    answer: NotRequired[str | None]


class RAGOutputState(TypedDict):
    answer: str | None
    retrieved_chunks: list[RetrievedChunkState]
    evaluation: RetrievalEvaluationState
    retry_count: int
    diagnosis: NotRequired[DiagnosisState]
    recovery_decision: NotRequired[RecoveryDecisionState]
