"""TF-IDF and count vectorization utilities for NLP feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)


class TFIDFVectorizerManager:
    """Manage TF-IDF vectorizer lifecycle for training and inference."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        """Fit vectorizer on texts and return sparse TF-IDF matrix."""
        logger.info("Fitting TF-IDF vectorizer and transforming training data.")
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        """Transform texts using an already fitted vectorizer."""
        logger.info("Transforming texts using fitted TF-IDF vectorizer.")
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, file_path: str | Path) -> None:
        """Persist fitted vectorizer to disk using joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info("TF-IDF vectorizer saved to %s", path)

    def load_vectorizer(self, file_path: str | Path) -> TfidfVectorizer:
        """Load a fitted vectorizer from disk and attach it to manager."""
        path = Path(file_path)
        self.vectorizer = joblib.load(path)
        logger.info("TF-IDF vectorizer loaded from %s", path)
        return self.vectorizer


def build_tfidf_vectorizer(
    corpus: Iterable[str], max_features: int = 5000
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Backward-compatible helper for fitting TF-IDF features."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    return vec, matrix


def build_count_vectorizer(
    corpus: Iterable[str], max_features: int = 2000
) -> Tuple[CountVectorizer, csr_matrix]:
    """Fit a count vectorizer, useful for topic modeling workflows."""
    vec = CountVectorizer(max_features=max_features, stop_words="english")
    matrix = vec.fit_transform(corpus)
    return vec, matrix
