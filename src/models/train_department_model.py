"""Training pipeline for department classification on civic grievances."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_macro_f1,
    calculate_precision,
    calculate_recall,
    generate_classification_report,
    plot_confusion_matrix,
)
from src.features.vectorize_tfidf import TFIDFVectorizerManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DepartmentModelTrainer:
    """Train and evaluate multiple baseline models for department classification."""

    def __init__(
        self,
        dataset_path: str | Path = "data/interim/cleaned_data.csv",
        model_output_path: str | Path = "artifacts/models/best_department_model.pkl",
        vectorizer_output_path: str | Path = "artifacts/vectorizers/tfidf_vectorizer.pkl",
        encoder_output_path: str | Path = "artifacts/encoders/department_label_encoder.pkl",
        random_state: int = 42,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_output_path = Path(model_output_path)
        self.vectorizer_output_path = Path(vectorizer_output_path)
        self.encoder_output_path = Path(encoder_output_path)
        self.random_state = random_state

        self.required_columns = ["processed_text", "department"]
        self.vectorizer_manager = TFIDFVectorizerManager(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.label_encoder = LabelEncoder()

    def _load_and_validate_dataset(self) -> pd.DataFrame:
        """Load dataset, validate columns, and remove null rows."""
        logger.info("Loading dataset from %s", self.dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        missing_columns = [c for c in self.required_columns if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        before_rows = len(df)
        df = df[self.required_columns].dropna().copy()
        df["processed_text"] = df["processed_text"].astype(str)
        df["department"] = df["department"].astype(str)
        logger.info("Removed %s rows with null values.", before_rows - len(df))

        if df.empty:
            raise ValueError("No data available after null removal.")

        return df

    def _define_models(self) -> dict[str, Any]:
        """Return model zoo for baseline comparison."""
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Multinomial Naive Bayes": MultinomialNB(),
            "Linear SVM": LinearSVC(random_state=self.random_state),
        }

    def train(self) -> dict[str, Any]:
        """Execute end-to-end training, evaluation, CV, and artifact saving."""
        try:
            df = self._load_and_validate_dataset()
            X = df["processed_text"]
            y = df["department"]

            y_encoded = self.label_encoder.fit_transform(y)
            class_names = list(self.label_encoder.classes_)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_encoded,
            )
            logger.info("Train size: %s | Test size: %s", len(X_train), len(X_test))

            X_train_tfidf = self.vectorizer_manager.fit_transform(X_train)
            X_test_tfidf = self.vectorizer_manager.transform(X_test)

            models = self._define_models()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            results: list[dict[str, Any]] = []
            best_model_name = ""
            best_model = None
            best_f1 = -1.0

            for model_name, model in models.items():
                logger.info("Training model: %s", model_name)
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)

                metrics = {
                    "model": model_name,
                    "accuracy": calculate_accuracy(y_test, y_pred),
                    "precision_macro": calculate_precision(y_test, y_pred),
                    "recall_macro": calculate_recall(y_test, y_pred),
                    "f1_macro": calculate_macro_f1(y_test, y_pred),
                    "classification_report": generate_classification_report(y_test, y_pred),
                }

                cv_scores = cross_val_score(
                    clone(model),
                    X_train_tfidf,
                    y_train,
                    cv=cv,
                    scoring="f1_macro",
                )
                metrics["cv_f1_macro_mean"] = float(cv_scores.mean())
                metrics["cv_f1_macro_std"] = float(cv_scores.std())

                results.append(metrics)
                logger.info(
                    "%s | F1_macro=%.4f | CV F1_macro=%.4f (+/- %.4f)",
                    model_name,
                    metrics["f1_macro"],
                    metrics["cv_f1_macro_mean"],
                    metrics["cv_f1_macro_std"],
                )

                plot_confusion_matrix(
                    y_true=y_test,
                    y_pred=y_pred,
                    labels=list(range(len(class_names))),
                    title=f"{model_name} - Confusion Matrix",
                    save_path=f"reports/{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
                )

                if metrics["f1_macro"] > best_f1:
                    best_f1 = metrics["f1_macro"]
                    best_model_name = model_name
                    best_model = model

            if best_model is None:
                raise RuntimeError("Best model selection failed.")

            self._save_artifacts(best_model)
            logger.info("Best model selected: %s (Macro F1: %.4f)", best_model_name, best_f1)

            return {
                "best_model_name": best_model_name,
                "best_model_f1_macro": best_f1,
                "results": results,
                "classes": class_names,
            }
        except Exception as exc:
            logger.exception("Training pipeline failed: %s", exc)
            raise

    def _save_artifacts(self, best_model: Any) -> None:
        """Persist model, TF-IDF vectorizer, and label encoder."""
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder_output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_model, self.model_output_path)
        self.vectorizer_manager.save_vectorizer(self.vectorizer_output_path)
        joblib.dump(self.label_encoder, self.encoder_output_path)

        logger.info("Saved model to %s", self.model_output_path)
        logger.info("Saved vectorizer to %s", self.vectorizer_output_path)
        logger.info("Saved label encoder to %s", self.encoder_output_path)


if __name__ == "__main__":
    trainer = DepartmentModelTrainer()
    summary = trainer.train()
    logger.info("Training completed. Summary: %s", summary["best_model_name"])
