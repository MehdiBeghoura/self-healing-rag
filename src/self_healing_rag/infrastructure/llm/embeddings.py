from ollama import Client

from self_healing_rag.config import settings


class OllamaEmbeddingClient:
    def __init__(self) -> None:
        self._client = Client(host=settings.ollama_host)
        self._model = settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embed(
            model=self._model,
            input=texts,
        )

        return response["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embed(
            model=self._model,
            input=text,
        )

        return response["embeddings"][0]
