from run_answer_evaluation import (
    evaluate_response,
    print_result,
    print_summary,
    validate_dataset,
)

from self_healing_rag.infrastructure.database.session import engine
from self_healing_rag.main import build_application_graph

EXPERIMENTAL_THRESHOLD = 0.4
TOP_K = 3


def main() -> int:
    try:
        benchmark_queries = validate_dataset()

        print(f"Experimental threshold: {EXPERIMENTAL_THRESHOLD}")
        print(
            f"Running {len(benchmark_queries)} benchmark queries..."
        )
        print(
            "This experiment does not change the production configuration."
        )

        results = []

        graph = build_application_graph()

        for index, benchmark_query in enumerate(
            benchmark_queries,
            start=1,
        ):
            output = graph.invoke(
                {
                    "query": benchmark_query.query,
                    "top_k": TOP_K,
                    "min_relevance_score": EXPERIMENTAL_THRESHOLD,
                }
            )

            # Evaluate the actual response produced by the graph.
            evaluation_result = evaluate_response(
                benchmark_query,
                output,
            )
            results.append(evaluation_result)

            print_result(
                index=index,
                total=len(benchmark_queries),
                result=evaluation_result,
            )

        print("\n=== Threshold Experiment Summary ===")
        print_summary(results)

        return (
            0
            if all(result.passed for result in results)
            else 1
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())