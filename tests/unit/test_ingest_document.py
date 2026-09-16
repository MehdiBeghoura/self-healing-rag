from uuid import uuid4

import pytest

from self_healing_rag.application.use_cases.ingest_document import (
    IngestDocumentRequest,
    IngestDocumentUseCase,
)


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.received_texts = texts
        return [[0.1] * 1024 for _ in texts]


class FakeDocumentStore:
    def __init__(self) -> None:
        self.received_title: str | None = None
        self.received_source: str | None = None
        self.received_chunks = None

    def __call__(
        self,
        title: str,
        source: str,
        chunks: list[tuple[int, str, list[float]]],
    ):
        self.received_title = title
        self.received_source = source
        self.received_chunks = chunks
        return uuid4()


def test_ingest_document_coordinates_chunking_embedding_and_storage() -> None:
    embeddings = FakeEmbeddingProvider()
    store = FakeDocumentStore()

    use_case = IngestDocumentUseCase(
        embedding_provider=embeddings,
        document_store=store,
    )

    document_id = use_case.execute(
        IngestDocumentRequest(
            title="Test document",
            source="test.txt",
            content="This is a small document that will be chunked.",
        )
    )

    assert document_id is not None
    assert store.received_title == "Test document"
    assert store.received_source == "test.txt"
    assert len(store.received_chunks) > 0
    assert len(embeddings.received_texts) == len(store.received_chunks)
    assert all(len(embedding) == 1024 for _, _, embedding in store.received_chunks)


def test_ingest_document_rejects_empty_content() -> None:
    use_case = IngestDocumentUseCase(
        embedding_provider=FakeEmbeddingProvider(),
        document_store=FakeDocumentStore(),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        use_case.execute(
            IngestDocumentRequest(
                title="Empty",
                source="empty.txt",
                content="   ",
            )
        )
