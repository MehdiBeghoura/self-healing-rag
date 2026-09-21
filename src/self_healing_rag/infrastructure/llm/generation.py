from ollama import Client

from self_healing_rag.config import settings


class OllamaGenerationClient:
    def __init__(self) -> None:
        self._client = Client(host=settings.ollama_host)
        self._model = settings.generation_model

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a question-answering assistant. "
                        "Answer using only the provided context. "
                        "If the context does not contain enough information "
                        "to answer the question, say that the provided "
                        "context is insufficient. Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"Context:\n{context}\n\nQuestion:\n{query}"),
                },
            ],
            options={
                "temperature": 0,
            },
        )

        return response.message.content
