from self_healing_rag.application.ports.retrieval import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context was retrieved."

    sections: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[Context {index}]\n"
            f"Title: {chunk.title}\n"
            f"Source: {chunk.source}\n"
            f"Content:\n{chunk.content}"
        )

    return "\n\n".join(sections)
