from ollama import Client
from pydantic import BaseModel

from self_healing_rag.application.ports.diagnosis import DiagnosisRequest
from self_healing_rag.config import settings
from self_healing_rag.domain.diagnosis import Diagnosis
from self_healing_rag.domain.recovery import RecoveryAction


class DiagnosisResponse(BaseModel):
    explanation: str
    recommended_action: RecoveryAction
    rewritten_query: str | None


class OllamaDiagnosticClient:
    def __init__(self) -> None:
        self._client = Client(host=settings.ollama_host)
        self._model = settings.generation_model

    def diagnose(self, request: DiagnosisRequest) -> Diagnosis:
        retrieved_context = "\n\n".join(
            (
                f"[Document {index}]\n"
                f"Title: {chunk.title}\n"
                f"Score: {chunk.score:.4f}\n"
                f"Content: {chunk.content}"
            )
            for index, chunk in enumerate(request.retrieved_chunks, start=1)
        )

        if not retrieved_context:
            retrieved_context = "No documents were retrieved."

        response = self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval diagnostic assistant. "
                        "The application has already determined the failure "
                        "type. Explain the likely retrieval problem and "
                        "recommend only an allowed recovery action. "
                        "Never invent database or system state. "
                        "Allowed actions are REWRITE_QUERY and STOP. "
                        "Use REWRITE_QUERY when changing the query could "
                        "reasonably improve retrieval. "
                        "Use STOP when rewriting is not justified. "
                        "When recommending REWRITE_QUERY, provide a concise "
                        "rewritten query. Otherwise rewritten_query must be null."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query:\n{request.query}\n\n"
                        f"Failure type:\n{request.failure_type.value}\n\n"
                        f"Failure reason:\n{request.reason}\n\n"
                        f"Retrieved evidence:\n{retrieved_context}"
                    ),
                },
            ],
            format=DiagnosisResponse.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        parsed = DiagnosisResponse.model_validate_json(response.message.content)

        return Diagnosis(
            failure_type=request.failure_type,
            explanation=parsed.explanation,
            recommended_action=parsed.recommended_action,
            rewritten_query=parsed.rewritten_query,
        )
