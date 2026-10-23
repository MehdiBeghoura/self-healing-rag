from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from self_healing_rag.api.routes.health import router as health_router
from self_healing_rag.api.routes.query import router as query_router
from self_healing_rag.application.use_cases.evaluate_retrieval import (
    EvaluateRetrievalUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsUseCase,
)
from self_healing_rag.config import settings
from self_healing_rag.infrastructure.database.session import engine
from self_healing_rag.infrastructure.llm.diagnosis import (
    OllamaDiagnosticClient,
)
from self_healing_rag.infrastructure.llm.embeddings import (
    OllamaEmbeddingClient,
)
from self_healing_rag.infrastructure.llm.generation import (
    OllamaGenerationClient,
)
from self_healing_rag.infrastructure.observability.tracing import (
    get_tracer_provider,
)
from self_healing_rag.infrastructure.retrieval.pgvector_retriever import (
    PgVectorRetriever,
)
from self_healing_rag.orchestration.rag_graph import build_rag_graph


def build_application_graph() -> Any:
    embedding_client = OllamaEmbeddingClient()
    diagnostic_client = OllamaDiagnosticClient()
    generation_client = OllamaGenerationClient()
    retriever = PgVectorRetriever()

    retrieval_use_case = RetrieveDocumentsUseCase(
        embedding_provider=embedding_client,
        retriever=retriever,
    )

    evaluation_use_case = EvaluateRetrievalUseCase()

    return build_rag_graph(
        retrieval_use_case=retrieval_use_case,
        evaluation_use_case=evaluation_use_case,
        diagnostic_provider=diagnostic_client,
        generation_provider=generation_client,
        max_retries=2,
    )


def create_app(
    graph_factory: Callable[[], Any] = build_application_graph,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.rag_graph = graph_factory()
            yield
        finally:
            if hasattr(app.state, "rag_graph"):
                del app.state.rag_graph

            engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(query_router)

    # Automatically create OpenTelemetry spans for incoming HTTP requests.
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=get_tracer_provider(),
    )

    return app


app = create_app()
