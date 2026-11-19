"""Tests for SimilarityX similarity engines."""

import pytest

from core.preprocessing import normalize_text
from core.tfidf_model import calculate_tfidf_similarity
from core.semantic_model import calculate_semantic_similarity
from utils.similarity import classify_similarity, compare_texts


def test_normalize_text_lowercases_and_removes_punctuation():
    assert normalize_text("  HELLO,   World!!!  ") == "hello world"


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        normalize_text("   ")

    with pytest.raises(ValueError, match="string"):
        normalize_text(None)


def test_tfidf_identical_sentences():
    assert calculate_tfidf_similarity(
        "How do I reset my password?",
        "How do I reset my password?",
    ) == 100.0


def test_tfidf_different_sentences():
    score = calculate_tfidf_similarity(
        "How do I reset my password?",
        "The weather is sunny in Madrid.",
    )
    assert 0.0 <= score < 100.0


def test_tfidf_punctuation_only_difference():
    score = calculate_tfidf_similarity(
        "How do I reset my password?",
        "How do I reset my password",
    )
    assert score == 100.0


def test_tfidf_empty_vocabulary():
    assert calculate_tfidf_similarity("hello", "!!!") == 0.0


def test_classification_thresholds():
    assert classify_similarity(85.0) == "Duplicate"
    assert classify_similarity(84.99) == "Similar"
    assert classify_similarity(60.0) == "Similar"
    assert classify_similarity(59.99) == "Different"


def test_compare_texts_invalid_threshold(monkeypatch):
    monkeypatch.setattr(
        "utils.similarity.calculate_tfidf_similarity",
        lambda text_a, text_b: 100.0,
    )
    monkeypatch.setattr(
        "utils.similarity.calculate_semantic_similarity",
        lambda text_a, text_b: 100.0,
    )

    with pytest.raises(ValueError, match="Thresholds"):
        compare_texts("hello", "hello", duplicate_threshold=50.0)


def test_compare_texts_runs_both_methods(monkeypatch):
    monkeypatch.setattr(
        "utils.similarity.calculate_tfidf_similarity",
        lambda text_a, text_b: 100.0,
    )
    monkeypatch.setattr(
        "utils.similarity.calculate_semantic_similarity",
        lambda text_a, text_b: 91.5,
    )

    result = compare_texts("hello", "hello")

    assert result["tfidf_score"] == 100.0
    assert result["semantic_score"] == 91.5
    assert result["classification"] == "Duplicate"


def test_compare_texts_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        compare_texts("", "A valid sentence")


def test_semantic_similarity_is_testable_without_fake_scores(monkeypatch):
    """Verify the model integration path with a real interface substitute."""
    import core.semantic_model as semantic_model

    class FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            import numpy as np

            return np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic_model, "get_semantic_model", lambda: FakeEmbeddingModel())

    assert semantic_model.calculate_semantic_similarity(
        "A first sentence",
        "A second sentence",
    ) == 100.0


@pytest.mark.parametrize(
    ("text_a", "text_b"),
    [
        ("How do I reset my password?", ""),
        ("", "How do I reset my password?"),
        ("!!!", "???"),
    ],
)
def test_only_empty_or_punctuation_input_is_rejected(text_a, text_b):
    with pytest.raises(ValueError):
        compare_texts(text_a, text_b)


def test_punctuation_difference_has_full_tfidf_overlap():
    assert calculate_tfidf_similarity(
        "What is the capital of France?",
        "what is the capital of france",
    ) == 100.0


def test_long_sentences_return_bounded_scores():
    text_a = (
        "Machine learning systems use training data to identify patterns, "
        "make predictions, and improve their behavior over repeated observations."
    )
    text_b = (
        "Machine learning models process representative datasets, detect patterns, "
        "and generate forecasts while learning from previous examples."
    )

    tfidf_score = calculate_tfidf_similarity(text_a, text_b)
    assert 0.0 <= tfidf_score <= 100.0

    semantic_score = calculate_semantic_similarity(text_a, text_b)
    assert 0.0 <= semantic_score <= 100.0


def test_repeated_comparisons_return_the_same_real_scores():
    text_a = "How do I reset my password?"
    text_b = "I forgot my password and need to reset it."

    first = compare_texts(text_a, text_b)
    second = compare_texts(text_a, text_b)

    assert first == second
    assert set(first) == {"tfidf_score", "semantic_score", "classification"}


def test_real_semantic_unrelated_sentences():
    score = calculate_semantic_similarity(
        "How do I reset my password?",
        "Quantum mechanics describes particle behavior.",
    )
    assert 0.0 <= score <= 100.0
