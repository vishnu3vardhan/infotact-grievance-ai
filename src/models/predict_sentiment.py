"""Sentiment inference utilities for the Week 3 model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from src.preprocessing.inference import normalize_texts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentPrediction:
    """Container for a single sentiment prediction."""

    sentiment: str
    confidence: float


class SentimentPredictor:
    """Load the trained sentiment model, vectorizer, and label encoder."""

    def __init__(
        self,
        model_path: str | Path = "artifacts/models/best_sentiment_model.pkl",
        vectorizer_path: str | Path = "artifacts/vectorizers/sentiment_tfidf_vectorizer.pkl",
        encoder_path: str | Path = "artifacts/encoders/sentiment_label_encoder.pkl",
    ) -> None:
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.encoder_path = Path(encoder_path)

        for path in (self.model_path, self.vectorizer_path, self.encoder_path):
            if not path.exists():
                raise FileNotFoundError(f"Required sentiment artifact not found: {path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_encoder = joblib.load(self.encoder_path)

        logger.info("Loaded sentiment artifacts from %s", self.model_path.parent)

    def _vectorize(self, texts: Sequence[str]):
        normalized = normalize_texts(texts)
        return self.vectorizer.transform(normalized)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities when supported by the estimator."""
        features = self._vectorize(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)
            scores = np.atleast_2d(scores)
            shifted = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

        raise AttributeError("Sentiment model does not expose probability or decision scores.")

    def predict(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        """Predict sentiments and confidences for a batch of texts."""
        probabilities = self.predict_proba(texts)
        class_indices = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        sentiments = self.label_encoder.inverse_transform(class_indices)

        return [
            SentimentPrediction(sentiment=str(sentiment), confidence=float(confidence))
            for sentiment, confidence in zip(sentiments, confidences, strict=False)
        ]

    def predict_one(self, text: str) -> SentimentPrediction:
        """Predict sentiment for a single text."""
        return self.predict([text])[0]