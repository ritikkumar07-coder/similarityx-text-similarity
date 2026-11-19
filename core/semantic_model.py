"""Sentence Transformer semantic similarity implementation."""

from functools import lru_cache
from typing import Any
import math

import numpy as np
from sentence_transformers import SentenceTransformer

from core.preprocessing import normalize_text


MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticModelError(RuntimeError):
    """Raised when the sentence embedding model cannot be loaded or used."""


@lru_cache(maxsize=1)
def get_semantic_model() -> SentenceTransformer:
    """Load and cache the configured sentence embedding model."""
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        raise SemanticModelError(
            f"Unable to load semantic model '{MODEL_NAME}'. "
            "Check the internet connection and model cache."
        ) from exc


def calculate_semantic_similarity(text_a: Any, text_b: Any) -> float:
    """Calculate cosine similarity between sentence embeddings as a percentage."""
    try:
        normalized_a = normalize_text(text_a)
        normalized_b = normalize_text(text_b)
    except ValueError:
        return 0.0

    if normalized_a == normalized_b:
        return 100.0

    try:
        model = get_semantic_model()
        embeddings = model.encode(
            [normalized_a, normalized_b],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except SemanticModelError:
        raise
    except Exception as exc:
        raise SemanticModelError("Unable to generate sentence embeddings.") from exc

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != 2 or embeddings.ndim != 2:
        raise SemanticModelError("Unexpected embedding output.")

    score = float(np.dot(embeddings[0], embeddings[1]))
    if not math.isfinite(score):
        raise SemanticModelError("Similarity computation returned a non-finite value.")
    score = max(-1.0, min(1.0, score))
    return round(max(0.0, score) * 100.0, 2)
