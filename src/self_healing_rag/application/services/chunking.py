from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str


def split_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = " ".join(text.split())

    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))

        chunks.append(
            TextChunk(
                index=index,
                content=normalized[start:end],
            )
        )

        if end == len(normalized):
            break

        start = end - overlap
        index += 1

    return chunks
