from self_healing_rag.infrastructure.llm.embeddings import OllamaEmbeddingClient


def main() -> None:
    client = OllamaEmbeddingClient()

    document_embeddings = client.embed_documents(
        [
            "PostgreSQL stores vector embeddings using pgvector.",
            "Python is a programming language.",
        ]
    )

    query_embedding = client.embed_query("How can PostgreSQL store vector embeddings?")

    print(f"Document embeddings: {len(document_embeddings)}")
    print(f"Document dimension: {len(document_embeddings[0])}")
    print(f"Query dimension: {len(query_embedding)}")


if __name__ == "__main__":
    main()
