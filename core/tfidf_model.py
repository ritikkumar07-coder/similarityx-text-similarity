"""Traditional TF-IDF similarity implementation."""

from typing import Any
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.preprocessing import normalize_text


def calculate_tfidf_similarity(text_a: Any, text_b: Any) -> float:
    """Calculate cosine similarity from TF-IDF vectors as a percentage."""
    try:
        normalized_a = normalize_text(text_a)
        normalized_b = normalize_text(text_b)
    except ValueError:
        return 0.0

    if normalized_a == normalized_b:
        return 100.0

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([normalized_a, normalized_b])
        if matrix.shape[1] == 0:
            return 0.0

        score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0

    if not math.isfinite(score):
        return 0.0
    return round(max(0.0, min(score, 1.0)) * 100.0, 2)


def compute_tfidf_similarity(text_a: Any, text_b: Any) -> float:
    """Alias for callers that prefer a compute-style function name."""
    return calculate_tfidf_similarity(text_a, text_b)


def tfidf_similarity(text_a: Any, text_b: Any) -> float:
    """Short alias for TF-IDF similarity."""
    return calculate_tfidf_similarity(text_a, text_b)
