import pytest

from self_healing_rag.domain.failures import FailureType
from self_healing_rag.domain.retrieval_evaluation import evaluate_retrieval


def test_empty_retrieval_is_no_documents():
    result = evaluate_retrieval(
        scores=[],
        min_relevance_score=0.5,
    )

    assert result.success is False
    assert result.failure_type == FailureType.NO_DOCUMENTS
    assert result.document_count == 0
    assert result.top_score is None


def test_retrieval_with_relevant_document_succeeds():
    result = evaluate_retrieval(
        scores=[0.82, 0.71, 0.64],
        min_relevance_score=0.6,
    )

    assert result.success is True
    assert result.failure_type is None
    assert result.document_count == 3
    assert result.top_score == 0.82


def test_low_top_score_is_low_relevance():
    result = evaluate_retrieval(
        scores=[0.42, 0.38, 0.31],
        min_relevance_score=0.6,
    )

    assert result.success is False
    assert result.failure_type == FailureType.LOW_RELEVANCE
    assert result.document_count == 3
    assert result.top_score == 0.42


def test_score_equal_to_threshold_is_accepted():
    result = evaluate_retrieval(
        scores=[0.6],
        min_relevance_score=0.6,
    )

    assert result.success is True
    assert result.failure_type is None


def test_valid_negative_cosine_similarity_is_accepted():
    result = evaluate_retrieval(
        scores=[-0.3],
        min_relevance_score=-0.5,
    )

    assert result.success is True
    assert result.top_score == -0.3


@pytest.mark.parametrize(
    "threshold",
    [
        -1.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_relevance_threshold_is_rejected(threshold: float):
    with pytest.raises(
        ValueError,
        match="min_relevance_score must be finite and between -1 and 1",
    ):
        evaluate_retrieval(
            scores=[0.5],
            min_relevance_score=threshold,
        )


@pytest.mark.parametrize(
    "score",
    [
        -1.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_retrieval_score_is_rejected(score: float):
    with pytest.raises(
        ValueError,
        match="retrieval scores must be finite and between -1 and 1",
    ):
        evaluate_retrieval(
            scores=[score],
            min_relevance_score=0.5,
        )