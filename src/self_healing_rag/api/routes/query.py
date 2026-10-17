import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from self_healing_rag.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("query must not be blank")

        return value


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any]


class RetrievalEvaluationResponse(BaseModel):
    success: bool
    failure_type: str | None
    reason: str
    document_count: int
    top_score: float | None


class DiagnosisResponse(BaseModel):
    failure_type: str
    explanation: str
    recommended_action: str
    rewritten_query: str | None


class RecoveryDecisionResponse(BaseModel):
    action: str
    rewritten_query: str | None
    reason: str


class QueryResponse(BaseModel):
    query: str
    retrieval_query: str
    answer: str | None
    retrieved_chunks: list[RetrievedChunkResponse]
    evaluation: RetrievalEvaluationResponse
    retry_count: int
    diagnosis: DiagnosisResponse | None = None
    recovery_decision: RecoveryDecisionResponse | None = None


def get_rag_graph(request: Request) -> Any:
    graph = getattr(request.app.state, "rag_graph", None)

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG workflow is not initialized.",
        )

    return graph


@router.post("/query", response_model=QueryResponse)
def query_rag(
    payload: QueryRequest,
    rag_graph: Annotated[Any, Depends(get_rag_graph)],
) -> QueryResponse:
    try:
        result = rag_graph.invoke(
            {
                "query": payload.query,
                "top_k": payload.top_k,
                "min_relevance_score": (
                    settings.retrieval_min_relevance_score
                ),
            }
        )
    except Exception as exc:
        logger.exception(
            "RAG workflow failed while processing a query"
        )

        raise HTTPException(
            status_code=500,
            detail="The RAG workflow could not process the request.",
        ) from exc

    return QueryResponse.model_validate(result)