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


class RAGInputState(TypedDict):
    query: str
    top_k: int
    min_relevance_score: float


class RAGState(RAGInputState):
    retrieved_chunks: NotRequired[list[RetrievedChunkState]]
    evaluation: NotRequired[RetrievalEvaluationState]
    answer: NotRequired[str | None]


class RAGOutputState(TypedDict):
    answer: str | None
    retrieved_chunks: list[RetrievedChunkState]
    evaluation: RetrievalEvaluationState
