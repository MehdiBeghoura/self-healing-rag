from uuid import UUID

from langgraph.graph import END, START, StateGraph

from self_healing_rag.application.ports.generation import GenerationProvider
from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.services.context_builder import build_context
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalRequest,
    EvaluateRetrievalUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
    RetrieveDocumentsUseCase,
)
from self_healing_rag.orchestration.state import (
    RAGInputState,
    RAGOutputState,
    RAGState,
    RetrievedChunkState,
)


def _chunk_to_state(chunk: RetrievedChunk) -> RetrievedChunkState:
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "content": chunk.content,
        "source": chunk.source,
        "title": chunk.title,
        "score": chunk.score,
        "metadata": chunk.metadata,
    }


def _chunk_from_state(chunk: RetrievedChunkState) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(chunk["chunk_id"]),
        document_id=UUID(chunk["document_id"]),
        content=chunk["content"],
        source=chunk["source"],
        title=chunk["title"],
        score=chunk["score"],
        metadata=chunk["metadata"],
    )


def build_rag_graph(
    retrieval_use_case: RetrieveDocumentsUseCase,
    evaluation_use_case: EvaluateRetrievalUseCase,
    generation_provider: GenerationProvider,
):
    builder = StateGraph(
        RAGState,
        input_schema=RAGInputState,
        output_schema=RAGOutputState,
    )

    def retrieve_node(state: RAGState) -> dict:
        chunks = retrieval_use_case.execute(
            RetrieveDocumentsRequest(
                query=state["query"],
                top_k=state["top_k"],
            )
        )

        return {"retrieved_chunks": [_chunk_to_state(chunk) for chunk in chunks]}

    def evaluate_node(state: RAGState) -> dict:
        chunks = [_chunk_from_state(chunk) for chunk in state["retrieved_chunks"]]

        evaluation = evaluation_use_case.execute(
            EvaluateRetrievalRequest(
                chunks=chunks,
                min_relevance_score=state["min_relevance_score"],
            )
        )

        return {
            "evaluation": {
                "success": evaluation.success,
                "failure_type": (
                    evaluation.failure_type.value
                    if evaluation.failure_type is not None
                    else None
                ),
                "reason": evaluation.reason,
                "document_count": evaluation.document_count,
                "top_score": evaluation.top_score,
            },
            "answer": None,
        }

    def generate_node(state: RAGState) -> dict:
        chunks = [_chunk_from_state(chunk) for chunk in state["retrieved_chunks"]]

        context = build_context(chunks)

        answer = generation_provider.generate(
            query=state["query"],
            context=context,
        )

        return {
            "answer": answer,
        }

    def route_after_evaluation(state: RAGState) -> str:
        if state["evaluation"]["success"]:
            return "generate"

        return "end"

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "end": END,
        },
    )

    builder.add_edge("generate", END)

    return builder.compile()
