from fastapi.testclient import TestClient

from self_healing_rag.main import create_app


class FakeRAGGraph:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def invoke(self, payload: dict) -> dict:
        self.inputs.append(payload)

        return {
            "query": payload["query"],
            "retrieval_query": payload["query"],
            "answer": "pgvector stores and searches vector embeddings.",
            "retrieved_chunks": [
                {
                    "chunk_id": (
                        "11111111-1111-4111-8111-111111111111"
                    ),
                    "document_id": (
                        "22222222-2222-4222-8222-222222222222"
                    ),
                    "title": "PostgreSQL Vector Guide",
                    "source": "test.md",
                    "content": (
                        "PostgreSQL can use pgvector to store vector "
                        "embeddings and perform similarity search."
                    ),
                    "score": 0.82,
                    "metadata": {},
                }
            ],
            "evaluation": {
                "success": True,
                "failure_type": None,
                "reason": (
                    "Retrieved documents meet the minimum relevance "
                    "threshold."
                ),
                "document_count": 1,
                "top_score": 0.82,
            },
            "retry_count": 0,
        }


def test_post_query_returns_rag_result() -> None:
    fake_graph = FakeRAGGraph()
    app = create_app(graph_factory=lambda: fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "What is pgvector used for?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is pgvector used for?"
    assert data["retrieval_query"] == "What is pgvector used for?"
    assert "pgvector" in data["answer"]
    assert data["evaluation"]["success"] is True
    assert data["retry_count"] == 0

    assert data["retrieved_chunks"][0]["title"] == (
        "PostgreSQL Vector Guide"
    )

    assert fake_graph.inputs == [
        {
            "query": "What is pgvector used for?",
            "top_k": 3,
            "min_relevance_score": 0.4,
        }
    ]


def test_post_query_rejects_blank_query() -> None:
    fake_graph = FakeRAGGraph()
    app = create_app(graph_factory=lambda: fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "   "},
        )

    assert response.status_code == 422
    assert fake_graph.inputs == []


def test_post_query_rejects_invalid_top_k() -> None:
    fake_graph = FakeRAGGraph()
    app = create_app(graph_factory=lambda: fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "What is pgvector used for?",
                "top_k": 0,
            },
        )

    assert response.status_code == 422
    assert fake_graph.inputs == []