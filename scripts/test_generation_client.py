from self_healing_rag.infrastructure.llm.generation import OllamaGenerationClient


def main() -> None:
    client = OllamaGenerationClient()

    answer = client.generate(
        query="What does PostgreSQL use pgvector for?",
        context=(
            "PostgreSQL can use the pgvector extension to store and "
            "search vector embeddings. Vector similarity can support "
            "semantic retrieval."
        ),
    )

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
