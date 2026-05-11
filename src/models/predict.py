"""Prediction utilities for department classification inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.inference import normalize_texts

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("artifacts/models/best_department_model.pkl")
DEFAULT_VECTORIZER_PATH = Path("artifacts/vectorizers/tfidf_vectorizer.pkl")
DEFAULT_ENCODER_PATH = Path("artifacts/encoders/department_label_encoder.pkl")
LEGACY_PIPELINE_PATH = Path("models/department_pipeline.joblib")


class DepartmentPredictor:
    """Load and serve department predictions from modern or legacy artifacts."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
        encoder_path: str | Path = DEFAULT_ENCODER_PATH,
        legacy_pipeline_path: str | Path = LEGACY_PIPELINE_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.encoder_path = Path(encoder_path)
        self.legacy_pipeline_path = Path(legacy_pipeline_path)

        self.model = None
        self.vectorizer = None
        self.label_encoder: LabelEncoder | None = None
        self.pipeline = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load modern artifact triplet, fallback to legacy single pipeline."""
        has_modern = self.model_path.exists() and self.vectorizer_path.exists() and self.encoder_path.exists()
        if has_modern:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.label_encoder = joblib.load(self.encoder_path)
            logger.info("Loaded modern department artifacts from artifacts/.")
            return

        if self.legacy_pipeline_path.exists():
            self.pipeline = joblib.load(self.legacy_pipeline_path)
            logger.info("Loaded legacy department pipeline from %s", self.legacy_pipeline_path)
            return

        raise FileNotFoundError(
            "No department artifacts found. Expected modern triplet under artifacts/ or legacy models/department_pipeline.joblib."
        )

    @staticmethod
    def _prepare_texts(texts: Iterable[str]) -> list[str]:
        return normalize_texts(texts)

    def _predict_modern(self, texts: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        return np.asarray(self.model.predict(features))

    def _predict_proba_modern(self, texts: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(features))
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(features))
            scores = np.atleast_2d(scores)
            shifted = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        raise AttributeError("Department model does not provide confidence scores.")

    def predict(self, texts: list[str]) -> list[str]:
        prepared_texts = self._prepare_texts(texts)
        if self.pipeline is not None:
            return list(self.pipeline.predict(prepared_texts))

        indices = self._predict_modern(prepared_texts)
        if self.label_encoder is None:
            raise RuntimeError("Department label encoder is not loaded.")
        labels = self.label_encoder.inverse_transform(indices.astype(int))
        return [str(label) for label in labels]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        prepared_texts = self._prepare_texts(texts)
        if self.pipeline is not None:
            if hasattr(self.pipeline, "predict_proba"):
                return np.asarray(self.pipeline.predict_proba(prepared_texts))
            clf = self.pipeline.named_steps.get("clf")
            vec = self.pipeline.named_steps.get("tfidf")
            if clf is None or vec is None:
                raise RuntimeError("Legacy pipeline does not have expected steps.")
            X = vec.transform(prepared_texts)
            if hasattr(clf, "predict_proba"):
                return np.asarray(clf.predict_proba(X))
            if hasattr(clf, "decision_function"):
                scores = np.asarray(clf.decision_function(X))
                scores = np.atleast_2d(scores)
                shifted = scores - scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(shifted)
                return exp_scores / exp_scores.sum(axis=1, keepdims=True)
            raise AttributeError("Legacy classifier does not provide confidence scores.")

        return self._predict_proba_modern(prepared_texts)


def predict_department(text: str) -> str:
    """Convenience function for single-text inference."""
    predictor = DepartmentPredictor()
    return predictor.predict([text])[0]

