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
                        "Treat the provided context as your only source "
                        "of evidence. Answer only when the context "
                        "explicitly states or directly supports the answer. "
                        "A missing statement is not evidence that a claim "
                        "is false. If the context does not establish "
                        "whether a claim is true or false, do not answer "
                        "yes or no. Instead, clearly state that the "
                        "provided context is insufficient to answer "
                        "the question. If the context explicitly "
                        "contradicts a claim, explain the contradiction "
                        "using only the provided context. Do not rely "
                        "on outside knowledge, infer unsupported "
                        "relationships, or invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{query}",
                },
            ],
            options={
                "temperature": 0,
            },
        )

        return response.message.content