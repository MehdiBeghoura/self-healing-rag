from uuid import UUID

from langgraph.graph import END, START, StateGraph

from self_healing_rag.application.ports.diagnosis import (
    DiagnosisRequest,
    DiagnosticProvider,
)
from self_healing_rag.application.ports.generation import GenerationProvider
from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.services.context_builder import build_context
from self_healing_rag.application.use_cases.decide_recovery import (
    DecideRecoveryRequest,
    DecideRecoveryUseCase,
)
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalRequest,
    EvaluateRetrievalUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
    RetrieveDocumentsUseCase,
)
from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.failures import FailureType
from self_healing_rag.domain.recovery import RecoveryAction, RecoveryDecision
from self_healing_rag.orchestration.state import (
    DiagnosisState,
    RAGInputState,
    RAGOutputState,
    RAGState,
    RecoveryDecisionState,
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


def _diagnosis_to_state(diagnosis: Diagnosis) -> DiagnosisState:
    return {
        "failure_type": diagnosis.failure_type.value,
        "explanation": diagnosis.explanation,
        "recommended_action": diagnosis.recommended_action.value,
        "rewritten_query": diagnosis.rewritten_query,
    }


def _diagnosis_from_state(diagnosis: DiagnosisState) -> Diagnosis:
    return Diagnosis(
        failure_type=FailureType(diagnosis["failure_type"]),
        explanation=diagnosis["explanation"],
        recommended_action=RecoveryAction(diagnosis["recommended_action"]),
        rewritten_query=diagnosis["rewritten_query"],
    )


def _recovery_decision_to_state(
    decision: RecoveryDecision,
) -> RecoveryDecisionState:
    return {
        "action": decision.action.value,
        "rewritten_query": decision.rewritten_query,
        "reason": decision.reason,
    }


def build_rag_graph(
    retrieval_use_case: RetrieveDocumentsUseCase,
    evaluation_use_case: EvaluateRetrievalUseCase,
    diagnostic_provider: DiagnosticProvider,
    generation_provider: GenerationProvider,
    max_retries: int = 2,
):
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    decide_recovery_use_case = DecideRecoveryUseCase()

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

        return {
            "retrieved_chunks": [_chunk_to_state(chunk) for chunk in chunks],
            "retry_count": state.get("retry_count", 0),
        }

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

    def diagnose_node(state: RAGState) -> dict:
        evaluation = state["evaluation"]

        if evaluation["failure_type"] is None:
            raise ValueError("A failed retrieval must have a failure type.")

        chunks = [_chunk_from_state(chunk) for chunk in state["retrieved_chunks"]]

        diagnosis = diagnostic_provider.diagnose(
            DiagnosisRequest(
                query=state["query"],
                failure_type=FailureType(evaluation["failure_type"]),
                reason=evaluation["reason"],
                retrieved_chunks=chunks,
            )
        )

        return {
            "diagnosis": _diagnosis_to_state(diagnosis),
        }

    def decide_recovery_node(state: RAGState) -> dict:
        diagnosis = _diagnosis_from_state(state["diagnosis"])

        decision = decide_recovery_use_case.execute(
            DecideRecoveryRequest(
                original_query=state["query"],
                diagnosis=diagnosis,
                retry_count=state.get("retry_count", 0),
                max_retries=max_retries,
            )
        )

        return {
            "recovery_decision": _recovery_decision_to_state(decision),
        }

    def apply_recovery_node(state: RAGState) -> dict:
        decision = state["recovery_decision"]

        if (
            decision["action"] != RecoveryAction.REWRITE_QUERY.value
            or not decision["rewritten_query"]
        ):
            raise ValueError("Apply recovery requires a valid query rewrite.")

        return {
            "query": decision["rewritten_query"],
            "retry_count": state.get("retry_count", 0) + 1,
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

        return "diagnose"

    def route_after_recovery(state: RAGState) -> str:
        if state["recovery_decision"]["action"] == RecoveryAction.REWRITE_QUERY.value:
            return "apply_recovery"

        return "end"

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("diagnose", diagnose_node)
    builder.add_node("decide_recovery", decide_recovery_node)
    builder.add_node("apply_recovery", apply_recovery_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "diagnose": "diagnose",
        },
    )

    builder.add_edge("diagnose", "decide_recovery")

    builder.add_conditional_edges(
        "decide_recovery",
        route_after_recovery,
        {
            "apply_recovery": "apply_recovery",
            "end": END,
        },
    )

    builder.add_edge("apply_recovery", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
