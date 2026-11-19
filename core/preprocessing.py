"""Text preprocessing utilities for SimilarityX."""

import re
import unicodedata
from typing import Any


def normalize_text(text: Any) -> str:
    """Return a safe, cleaned representation of text.

    Raises:
        ValueError: If the value is not text or contains no usable characters.
    """
    if not isinstance(text, str):
        raise ValueError("Text input must be a string.")

    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("Text input cannot be empty.")

    normalized = "".join(
        " "
        if unicodedata.category(character).startswith(("P", "S"))
        or unicodedata.category(character).startswith("C")
        else character
        for character in normalized
    )
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()

    if not normalized:
        raise ValueError("Text input must contain at least one alphanumeric character.")

    return normalized


def clean_text(text: Any) -> str:
    """Backward-friendly alias for text normalization."""
    return normalize_text(text)
