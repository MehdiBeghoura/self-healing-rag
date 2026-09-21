from uuid import uuid4

import pytest

from self_healing_rag.application.ports.retrieval import RetrievedChunk
from self_healing_rag.application.use_cases.retrieve_documents import (
    RetrieveDocumentsRequest,
    RetrieveDocumentsUseCase,
)


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.received_query: str | None = None

    def embed_query(self, text: str) -> list[float]:
        self.received_query = text
        return [0.1] * 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class FakeRetriever:
    def __init__(self) -> None:
        self.received_embedding: list[float] | None = None
        self.received_top_k: int | None = None

    def retrieve(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.received_embedding = embedding
        self.received_top_k = top_k

        return [
            RetrievedChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="PostgreSQL uses pgvector.",
                source="test.txt",
                title="Test",
                score=0.9,
                metadata={},
            )
        ]


def test_retrieve_documents_embeds_query_and_retrieves() -> None:
    embeddings = FakeEmbeddingProvider()
    retriever = FakeRetriever()

    use_case = RetrieveDocumentsUseCase(
        embedding_provider=embeddings,
        retriever=retriever,
    )

    results = use_case.execute(
        RetrieveDocumentsRequest(
            query="  How does PostgreSQL use vectors?  ",
            top_k=3,
        )
    )

    assert embeddings.received_query == "How does PostgreSQL use vectors?"
    assert retriever.received_embedding == [0.1] * 1024
    assert retriever.received_top_k == 3
    assert len(results) == 1


@pytest.mark.parametrize(
    "test_request",
    [
        RetrieveDocumentsRequest(query=""),
        RetrieveDocumentsRequest(query="   "),
        RetrieveDocumentsRequest(query="question", top_k=0),
        RetrieveDocumentsRequest(query="question", top_k=-1),
    ],
)
def test_retrieve_documents_rejects_invalid_requests(
    test_request: RetrieveDocumentsRequest,
) -> None:
    use_case = RetrieveDocumentsUseCase(
        embedding_provider=FakeEmbeddingProvider(),
        retriever=FakeRetriever(),
    )

    with pytest.raises(ValueError):
        use_case.execute(test_request)
