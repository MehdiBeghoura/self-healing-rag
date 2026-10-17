from dataclasses import dataclass
from statistics import mean
from uuid import UUID

from sqlalchemy import delete

from self_healing_rag.application.use_cases.ingest_document import (
    IngestDocumentRequest,
    IngestDocumentUseCase,
)
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
    RetrieveDocumentsUseCase,
)
from self_healing_rag.evaluation.dataset import (
    BENCHMARK_DOCUMENTS,
    BENCHMARK_QUERIES,
    QueryType,
)
from self_healing_rag.infrastructure.database.ingestion import save_document
from self_healing_rag.infrastructure.database.models import Document
from self_healing_rag.infrastructure.database.session import SessionLocal
from self_healing_rag.infrastructure.llm.embeddings import (
    OllamaEmbeddingClient,
)
from self_healing_rag.infrastructure.retrieval.pgvector_retriever import (
    PgVectorRetriever,
)


@dataclass(frozen=True)
class BenchmarkResult:
    query: str
    query_type: QueryType
    expected_titles: tuple[str, ...]
    retrieved_titles: list[str]
    retrieved_scores: list[float]
    top_score: float | None
    top1_relevant: bool
    top_k_relevant: bool
    best_relevant_score: float | None


def run_benchmark() -> list[BenchmarkResult]:
    embedding_client = OllamaEmbeddingClient()
    retriever = PgVectorRetriever()

    ingest_use_case = IngestDocumentUseCase(
        embedding_provider=embedding_client,
        document_store=save_document,
    )

    retrieve_use_case = RetrieveDocumentsUseCase(
        embedding_provider=embedding_client,
        retriever=retriever,
    )

    document_ids: list[UUID] = []

    try:
        print("Ingesting benchmark documents...")

        for document in BENCHMARK_DOCUMENTS:
            document_id = ingest_use_case.execute(
                IngestDocumentRequest(
                    title=document.title,
                    source=document.source,
                    content=document.content,
                )
            )
            document_ids.append(document_id)

        print(f"Inserted {len(document_ids)} benchmark documents.\n")

        results: list[BenchmarkResult] = []

        for benchmark_query in BENCHMARK_QUERIES:
            retrieved_chunks = retrieve_use_case.execute(
                RetrieveDocumentsRequest(
                    query=benchmark_query.query,
                    top_k=3,
                    document_ids=tuple(document_ids),
                )
            )

            retrieved_titles = [
                chunk.title for chunk in retrieved_chunks
            ]
            retrieved_scores = [
                chunk.score for chunk in retrieved_chunks
            ]

            expected_titles = set(benchmark_query.relevant_titles)
            retrieved_title_set = set(retrieved_titles)

            top1_relevant = bool(
                retrieved_titles
                and retrieved_titles[0] in expected_titles
            )

            top_k_relevant = bool(
                expected_titles & retrieved_title_set
            )

            relevant_scores = [
                chunk.score
                for chunk in retrieved_chunks
                if chunk.title in expected_titles
            ]

            best_relevant_score = (
                max(relevant_scores) if relevant_scores else None
            )

            results.append(
                BenchmarkResult(
                    query=benchmark_query.query,
                    query_type=benchmark_query.query_type,
                    expected_titles=benchmark_query.relevant_titles,
                    retrieved_titles=retrieved_titles,
                    retrieved_scores=retrieved_scores,
                    top_score=(
                        retrieved_scores[0]
                        if retrieved_scores
                        else None
                    ),
                    top1_relevant=top1_relevant,
                    top_k_relevant=top_k_relevant,
                    best_relevant_score=best_relevant_score,
                )
            )

        return results

    finally:
        # Delete only the documents inserted by this benchmark run.
        # PostgreSQL ON DELETE CASCADE removes their associated chunks.
        if document_ids:
            with SessionLocal.begin() as session:
                session.execute(
                    delete(Document).where(
                        Document.id.in_(document_ids)
                    )
                )


def print_results(results: list[BenchmarkResult]) -> None:
    print("=== Retrieval Calibration Results ===\n")

    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result.query}")
        print(f"Category: {result.query_type}")
        print(
            f"Expected: "
            f"{result.expected_titles or 'No relevant document'}"
        )

        if result.retrieved_titles:
            print("Retrieved:")

            for rank, (title, score) in enumerate(
                zip(
                    result.retrieved_titles,
                    result.retrieved_scores,
                    strict=True,
                ),
                start=1,
            ):
                print(f"  {rank}. {title} — score={score:.4f}")
        else:
            print("Retrieved: none")

        print(f"Top-1 relevant: {result.top1_relevant}")
        print(f"Top-k relevant: {result.top_k_relevant}")

        if result.best_relevant_score is not None:
            print(
                f"Best relevant score: "
                f"{result.best_relevant_score:.4f}"
            )
        else:
            print("Best relevant score: N/A")

        print()


def print_summary(results: list[BenchmarkResult]) -> None:
    query_types: tuple[QueryType, ...] = (
        "answerable",
        "hard_negative",
        "unrelated",
    )

    thresholds = (0.4, 0.5, 0.6, 0.7, 0.8)

    print("=== Retrieval Calibration Summary ===\n")

    # Report retrieval-quality metrics for answerable queries.
    answerable = [
        result
        for result in results
        if result.query_type == "answerable"
    ]

    answerable_count = len(answerable)

    if answerable_count:
        top1_hits = sum(
            result.top1_relevant for result in answerable
        )
        top_k_hits = sum(
            result.top_k_relevant for result in answerable
        )

        print(
            f"Answerable Top-1 relevance: "
            f"{top1_hits}/{answerable_count} "
            f"({top1_hits / answerable_count * 100:.1f}%)"
        )
        print(
            f"Answerable Top-k relevance: "
            f"{top_k_hits}/{answerable_count} "
            f"({top_k_hits / answerable_count * 100:.1f}%)"
        )
        print()
    else:
        print(
            "No answerable queries available for relevance metrics.\n"
        )

    # Show score distributions and threshold acceptance by category.
    for query_type in query_types:
        category_results = [
            result
            for result in results
            if result.query_type == query_type
        ]

        scores = [
            result.top_score
            for result in category_results
            if result.top_score is not None
        ]

        print(f"Category: {query_type}")
        print(f"Queries: {len(category_results)}")

        if scores:
            print(
                f"Top-score range: {min(scores):.4f} - "
                f"{max(scores):.4f}"
            )
            print(f"Average top score: {mean(scores):.4f}")
        else:
            print("No retrieval scores available.")

        print("Threshold acceptance:")

        for threshold in thresholds:
            accepted = sum(
                result.top_score is not None
                and result.top_score >= threshold
                for result in category_results
            )

            total = len(category_results)
            percentage = (
                accepted / total * 100 if total else 0.0
            )

            print(
                f"  {threshold:.1f}: "
                f"{accepted}/{total} ({percentage:.1f}%)"
            )

        print()


def main() -> None:
    results = run_benchmark()
    print_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()