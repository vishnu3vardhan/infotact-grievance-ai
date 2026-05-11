"""Inference-time text normalization helpers."""

from __future__ import annotations

import logging
import re
import string
from functools import lru_cache
from typing import Iterable, List

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_text_cleaner():
    """Load the project text cleaner lazily to avoid startup failures."""
    try:
        from src.preprocessing.cleaning import TextCleaner

        return TextCleaner()
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        logger.info("spaCy is not installed; using lightweight inference normalization.")
        return None
    except Exception as exc:  # pragma: no cover - fallback path for runtime safety
        logger.warning("Falling back to lightweight text normalization: %s", exc)
        return None


def basic_normalize_text(text: str) -> str:
    """Apply a lightweight normalization pipeline suitable for inference fallback."""
    normalized = str(text).lower()
    normalized = re.sub(r"http\S+|www\S+", "", normalized)
    normalized = re.sub(r"\S+@\S+", "", normalized)
    normalized = re.sub(r"\b\d{10}\b", "", normalized)
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\d+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_text(text: str) -> str:
    """Normalize a single piece of text using the project cleaner when available."""
    cleaner = _load_text_cleaner()
    if cleaner is None:
        return basic_normalize_text(text)

    try:
        return cleaner.preprocess_text(str(text))
    except Exception as exc:  # pragma: no cover - runtime safety fallback
        logger.warning("Cleaner preprocessing failed, using fallback normalization: %s", exc)
        return basic_normalize_text(text)


def normalize_texts(texts: Iterable[str]) -> List[str]:
    """Normalize a batch of input texts."""
    return [normalize_text(text) for text in texts]
