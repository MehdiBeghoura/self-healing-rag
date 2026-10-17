import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from self_healing_rag.evaluation.dataset import (
    BENCHMARK_QUERIES,
    BenchmarkQuery,
    QueryType,
)

BASE_URL = os.environ.get(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

TOP_K = 3
REQUEST_TIMEOUT_SECONDS = 180

ABSTENTION_PHRASES = (
    "insufficient",
    "does not contain",
    "do not contain",
    "not enough information",
    "not enough context",
    "cannot determine",
    "can't determine",
    "unable to answer",
    "not provided",
    "not supported by the context",
    "not in the context",
    "no information about",
)


@dataclass(frozen=True)
class EvaluationResult:
    query: str
    query_type: QueryType
    passed: bool
    detail: str
    answer: str | None
    top_score: float | None
    retrieval_success: bool | None


def validate_dataset() -> list[BenchmarkQuery]:
    """Validate evaluation metadata before sending API requests."""
    queries = list(BENCHMARK_QUERIES)

    if not queries:
        raise ValueError("The benchmark dataset contains no queries.")

    missing_criteria = [
        query.query
        for query in queries
        if query.query_type == "answerable"
        and not query.required_answer_concepts
    ]

    if missing_criteria:
        raise ValueError(
            "Answerable queries are missing required answer concepts: "
            + "; ".join(missing_criteria)
        )

    return queries


def call_query_api(
    benchmark_query: BenchmarkQuery,
) -> dict[str, Any]:
    """Submit one benchmark query to the running FastAPI application."""
    request_body = json.dumps(
        {
            "query": benchmark_query.query,
            "top_k": TOP_K,
        }
    ).encode("utf-8")

    request = Request(
        url=f"{BASE_URL}/query",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response_text = response.read().decode("utf-8")

    except HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"API returned HTTP {exc.code}: {error_body}"
        ) from exc

    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not reach {BASE_URL}. "
            "Check that the FastAPI server is running. "
            f"Details: {exc}"
        ) from exc

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The API returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(
            "The API returned an unexpected JSON response."
        )

    return payload


def is_abstention(answer: str | None) -> bool:
    """
    Detect common abstention phrases.

    This is a heuristic, not a guarantee that the complete response
    contains no unsupported claims.
    """
    if answer is None or not answer.strip():
        return True

    normalized_answer = answer.casefold()

    return any(
        phrase in normalized_answer
        for phrase in ABSTENTION_PHRASES
    )


def evaluate_response(
    benchmark_query: BenchmarkQuery,
    response: dict[str, Any],
) -> EvaluationResult:
    """Evaluate required answer concepts or expected abstention."""
    raw_evaluation = response.get("evaluation")

    evaluation = (
        raw_evaluation
        if isinstance(raw_evaluation, dict)
        else {}
    )

    raw_score = evaluation.get("top_score")

    top_score = (
        float(raw_score)
        if isinstance(raw_score, (int, float))
        and not isinstance(raw_score, bool)
        else None
    )

    raw_retrieval_success = evaluation.get("success")

    retrieval_success = (
        raw_retrieval_success
        if isinstance(raw_retrieval_success, bool)
        else None
    )

    raw_answer = response.get("answer")
    answer = raw_answer if isinstance(raw_answer, str) else None

    if benchmark_query.query_type == "answerable":
        if answer is None or not answer.strip():
            passed = False
            detail = (
                "Expected an answer, but the API returned no answer."
            )
        else:
            normalized_answer = answer.casefold()

            missing_concepts = [
                " / ".join(alternatives)
                for alternatives in (
                    benchmark_query.required_answer_concepts
                )
                if not any(
                    alternative.casefold() in normalized_answer
                    for alternative in alternatives
                )
            ]

            passed = not missing_concepts

            if passed:
                detail = "All required answer concepts were found."
            else:
                detail = (
                    "Missing required concepts: "
                    + "; ".join(missing_concepts)
                )

    else:
        passed = is_abstention(answer)

        if passed:
            detail = (
                "The system abstained according to the heuristic."
            )
        else:
            detail = (
                "Expected abstention, but the system returned "
                "a substantive answer."
            )

    return EvaluationResult(
        query=benchmark_query.query,
        query_type=benchmark_query.query_type,
        passed=passed,
        detail=detail,
        answer=answer,
        top_score=top_score,
        retrieval_success=retrieval_success,
    )


def print_result(
    index: int,
    total: int,
    result: EvaluationResult,
) -> None:
    """Print one evaluation result."""
    status = "PASS" if result.passed else "FAIL"

    print(f"\n[{index}/{total}] {result.query_type}: {result.query}")
    print(f"Result: {status}")
    print(
        "Retrieval evaluation passed: "
        f"{result.retrieval_success}"
    )

    if result.top_score is not None:
        print(f"Top retrieval score: {result.top_score:.4f}")

    print(f"Answer: {result.answer!r}")
    print(f"Details: {result.detail}")


def print_summary(results: list[EvaluationResult]) -> None:
    """Print overall results and category-specific totals."""
    passed_total = sum(result.passed for result in results)

    print("\n=== Answer Evaluation Summary ===")
    print(f"Overall: {passed_total}/{len(results)} passed")

    query_types: tuple[QueryType, ...] = (
        "answerable",
        "hard_negative",
        "unrelated",
    )

    for query_type in query_types:
        category_results = [
            result
            for result in results
            if result.query_type == query_type
        ]

        category_passed = sum(
            result.passed for result in category_results
        )

        total = len(category_results)

        percentage = (
            category_passed / total * 100
            if total
            else 0.0
        )

        print(
            f"{query_type}: {category_passed}/{total} "
            f"({percentage:.1f}%) passed"
        )


def main() -> int:
    try:
        benchmark_queries = validate_dataset()
    except ValueError as exc:
        print(f"Evaluation setup error: {exc}", file=sys.stderr)
        return 1

    results: list[EvaluationResult] = []

    print(f"API: {BASE_URL}")
    print(
        f"Running {len(benchmark_queries)} answer evaluation cases "
        "sequentially..."
    )

    for index, benchmark_query in enumerate(
        benchmark_queries,
        start=1,
    ):
        try:
            response = call_query_api(benchmark_query)
        except (RuntimeError, TypeError) as exc:
            print(f"\nEvaluation error: {exc}", file=sys.stderr)
            return 1

        result = evaluate_response(
            benchmark_query,
            response,
        )
        results.append(result)

        print_result(
            index=index,
            total=len(benchmark_queries),
            result=result,
        )

    print_summary(results)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())