"""Unified text similarity comparison engine."""

import math
from typing import Any, Dict

from core.preprocessing import normalize_text
from core.semantic_model import calculate_semantic_similarity
from core.tfidf_model import calculate_tfidf_similarity


def classify_similarity(
    semantic_score: float,
    duplicate_threshold: float = 85.0,
    similar_threshold: float = 60.0,
) -> str:
    """Classify a semantic percentage score using configurable thresholds."""
    if not isinstance(semantic_score, (int, float)) or not math.isfinite(float(semantic_score)):
        raise ValueError("Semantic score must be a finite number.")
    if not isinstance(duplicate_threshold, (int, float)) or not isinstance(similar_threshold, (int, float)):
        raise ValueError("Similarity thresholds must be numeric.")
    if not math.isfinite(float(duplicate_threshold)) or not math.isfinite(float(similar_threshold)):
        raise ValueError("Similarity thresholds must be finite numbers.")
    if not 0.0 <= similar_threshold <= duplicate_threshold <= 100.0:
        raise ValueError(
            "Thresholds must satisfy 0 <= similar_threshold "
            "<= duplicate_threshold <= 100."
        )

    if semantic_score >= duplicate_threshold:
        return "Duplicate"
    if semantic_score >= similar_threshold:
        return "Similar"
    return "Different"


def compare_texts(
    text_a: Any,
    text_b: Any,
    duplicate_threshold: float = 85.0,
    similar_threshold: float = 60.0,
) -> Dict[str, float | str]:
    """Run TF-IDF and semantic comparison and return a unified result."""
    normalized_a = normalize_text(text_a)
    normalized_b = normalize_text(text_b)

    tfidf_score = calculate_tfidf_similarity(normalized_a, normalized_b)
    semantic_score = calculate_semantic_similarity(normalized_a, normalized_b)
    classification = classify_similarity(
        semantic_score,
        duplicate_threshold=duplicate_threshold,
        similar_threshold=similar_threshold,
    )

    return {
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "classification": classification,
    }


def calculate_similarity(
    text_a: Any,
    text_b: Any,
    duplicate_threshold: float = 85.0,
    similar_threshold: float = 60.0,
) -> Dict[str, float | str]:
    """Alias for the unified comparison function."""
    return compare_texts(
        text_a,
        text_b,
        duplicate_threshold=duplicate_threshold,
        similar_threshold=similar_threshold,
    )
