import pytest

from self_healing_rag.application.services.chunking import split_text


def test_split_text_creates_ordered_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_text(
        text,
        chunk_size=10,
        overlap=2,
    )

    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content == "ijklmnopqr"
    assert chunks[2].content == "qrstuvwxyz"


def test_split_text_returns_empty_list_for_blank_text() -> None:
    assert split_text("   \n\n  ") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 1),
        (-1, 1),
        (10, 10),
        (10, 11),
    ],
)
def test_split_text_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        split_text(
            "some text",
            chunk_size=chunk_size,
            overlap=overlap,
        )
