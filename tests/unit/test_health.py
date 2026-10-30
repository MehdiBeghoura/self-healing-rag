from contextlib import nullcontext

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from self_healing_rag.main import create_app

app = create_app(
    graph_factory=lambda checkpointer: None,
    checkpointer_factory=lambda: nullcontext(InMemorySaver()),
)


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
