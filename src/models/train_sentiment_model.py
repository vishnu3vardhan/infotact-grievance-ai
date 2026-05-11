"""Training pipeline for sentiment classification on civic grievances."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_macro_f1,
    calculate_precision,
    calculate_recall,
    generate_classification_report,
    plot_confusion_matrix,
)
from src.features.vectorize_tfidf import TFIDFVectorizerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentTrainingConfig:
    """Configuration for sentiment model training."""

    dataset_path: Path = Path("data/interim/cleaned_data.csv")
    model_output_path: Path = Path("artifacts/models/best_sentiment_model.pkl")
    vectorizer_output_path: Path = Path("artifacts/vectorizers/sentiment_tfidf_vectorizer.pkl")
    encoder_output_path: Path = Path("artifacts/encoders/sentiment_label_encoder.pkl")
    comparison_output_path: Path = Path("reports/sentiment_model_comparison.csv")
    confusion_matrix_output_path: Path = Path("reports/sentiment_confusion_matrix.png")
    random_state: int = 42
    min_class_samples_for_cv: int = 5
    use_transformer: bool = False
    transformer_model_name: str = "distilbert-base-uncased"
    class_mapping: dict[str, str] = field(
        default_factory=lambda: {"Urgent": "Critical/Urgent", "Critical": "Critical/Urgent"}
    )


class SentimentModelTrainer:
    """Train and evaluate a multiclass sentiment classifier."""

    def __init__(self, config: SentimentTrainingConfig | None = None) -> None:
        self.config = config or SentimentTrainingConfig()
        self.required_columns = ["processed_text", "sentiment"]
        self.vectorizer_manager = TFIDFVectorizerManager(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self.label_encoder = LabelEncoder()

    def _load_and_validate_dataset(self) -> pd.DataFrame:
        logger.info("Loading sentiment dataset from %s", self.config.dataset_path)
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        df = pd.read_csv(self.config.dataset_path)
        missing_columns = [column for column in self.required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        before_rows = len(df)
        df = df[self.required_columns].dropna().copy()
        df["processed_text"] = df["processed_text"].astype(str)
        df["sentiment"] = df["sentiment"].astype(str).str.strip()
        df["sentiment"] = df["sentiment"].replace(self.config.class_mapping)

        if df.empty:
            raise ValueError("No sentiment data available after null removal.")

        logger.info("Loaded %s rows after cleaning (%s removed).", len(df), before_rows - len(df))
        return df

    def _balance_minority_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Upsample rare classes enough to support stratified split and cross-validation."""
        counts = df["sentiment"].value_counts()
        if counts.min() >= self.config.min_class_samples_for_cv:
            return df

        logger.warning(
            "Balancing rare sentiment classes to at least %s samples for stratified training.",
            self.config.min_class_samples_for_cv,
        )
        balanced_parts = [df]
        for label, count in counts.items():
            if count >= self.config.min_class_samples_for_cv:
                continue
            required = self.config.min_class_samples_for_cv - count
            class_rows = df[df["sentiment"] == label]
            sampled = resample(
                class_rows,
                replace=True,
                n_samples=required,
                random_state=self.config.random_state,
            )
            balanced_parts.append(sampled)

        balanced_df = pd.concat(balanced_parts, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1.0, random_state=self.config.random_state).reset_index(drop=True)
        return balanced_df

    def _define_models(self) -> dict[str, Any]:
        logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.config.random_state)
        calibrated_svm = CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=self.config.random_state),
            method="sigmoid",
            cv=3,
        )
        return {
            "Logistic Regression": logistic,
            "Calibrated Linear SVM": calibrated_svm,
            "Multinomial Naive Bayes": MultinomialNB(),
        }

    def _fit_transformer_path(self) -> None:
        """Optional transformer fine-tuning hook behind a config flag."""
        try:
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Transformer training requested, but the transformers package is not installed."
            ) from exc

        raise NotImplementedError(
            "Transformer fine-tuning is intentionally disabled in the default offline workflow."
        )

    def train(self) -> dict[str, Any]:
        """Train, evaluate, and persist the best sentiment model."""
        try:
            if self.config.use_transformer:
                self._fit_transformer_path()

            df = self._load_and_validate_dataset()
            df = self._balance_minority_classes(df)

            X = df["processed_text"]
            y = df["sentiment"]

            y_encoded = self.label_encoder.fit_transform(y)
            class_names = list(self.label_encoder.classes_)
            logger.info("Sentiment classes: %s", class_names)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=y_encoded,
            )
            logger.info("Train size: %s | Test size: %s", len(X_train), len(X_test))

            X_train_tfidf = self.vectorizer_manager.fit_transform(X_train)
            X_test_tfidf = self.vectorizer_manager.transform(X_test)

            class_distribution = pd.Series(y_train).value_counts()
            min_class_count = int(class_distribution.min())
            cv_splits = max(2, min(5, min_class_count))
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.config.random_state)

            results: list[dict[str, Any]] = []
            best_model_name = ""
            best_model = None
            best_f1 = -1.0

            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for model_name, model in self._define_models().items():
                logger.info("Training sentiment model: %s", model_name)
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

                if metrics["f1_macro"] > best_f1:
                    best_f1 = metrics["f1_macro"]
                    best_model_name = model_name
                    best_model = model

            if best_model is None:
                raise RuntimeError("Sentiment model selection failed.")

            self._save_artifacts(best_model)
            self._save_comparison_csv(results)

            best_predictions = best_model.predict(X_test_tfidf)
            best_test_labels = self.label_encoder.inverse_transform(y_test)
            best_pred_labels = self.label_encoder.inverse_transform(best_predictions)
            plot_confusion_matrix(
                y_true=best_test_labels,
                y_pred=best_pred_labels,
                labels=class_names,
                title=f"{best_model_name} - Sentiment Confusion Matrix",
                save_path=self.config.confusion_matrix_output_path,
            )

            best_summary = next(item for item in results if item["model"] == best_model_name)
            logger.info("Best sentiment model selected: %s (Macro F1: %.4f)", best_model_name, best_f1)

            return {
                "best_model_name": best_model_name,
                "best_model_f1_macro": best_f1,
                "best_model_metrics": best_summary,
                "results": results,
                "classes": class_names,
            }
        except Exception as exc:
            logger.exception("Sentiment training pipeline failed: %s", exc)
            raise

    def _save_artifacts(self, best_model: Any) -> None:
        """Persist model, TF-IDF vectorizer, and label encoder to the configured artifact paths."""
        self.config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.encoder_output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_model, self.config.model_output_path)
        self.vectorizer_manager.save_vectorizer(self.config.vectorizer_output_path)
        joblib.dump(self.label_encoder, self.config.encoder_output_path)

        logger.info("Saved sentiment model to %s", self.config.model_output_path)
        logger.info("Saved sentiment vectorizer to %s", self.config.vectorizer_output_path)
        logger.info("Saved sentiment label encoder to %s", self.config.encoder_output_path)

    def _save_comparison_csv(self, results: list[dict[str, Any]]) -> None:
        """Save a compact model comparison table for reporting."""
        self.config.comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df = pd.DataFrame(results).drop(columns=["classification_report"])
        comparison_df.to_csv(self.config.comparison_output_path, index=False)
        logger.info("Saved sentiment model comparison to %s", self.config.comparison_output_path)


def train_sentiment_classifier(
    input_csv: str | Path = "data/interim/cleaned_data.csv",
    model_output_path: str | Path = "artifacts/models/best_sentiment_model.pkl",
    vectorizer_output_path: str | Path = "artifacts/vectorizers/sentiment_tfidf_vectorizer.pkl",
    encoder_output_path: str | Path = "artifacts/encoders/sentiment_label_encoder.pkl",
    comparison_output_path: str | Path = "reports/sentiment_model_comparison.csv",
    confusion_matrix_output_path: str | Path = "reports/sentiment_confusion_matrix.png",
    use_transformer: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper for scripted sentiment training."""
    config = SentimentTrainingConfig(
        dataset_path=Path(input_csv),
        model_output_path=Path(model_output_path),
        vectorizer_output_path=Path(vectorizer_output_path),
        encoder_output_path=Path(encoder_output_path),
        comparison_output_path=Path(comparison_output_path),
        confusion_matrix_output_path=Path(confusion_matrix_output_path),
        use_transformer=use_transformer,
    )
    trainer = SentimentModelTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    summary = train_sentiment_classifier()
    logger.info("Sentiment training completed. Summary: %s", json.dumps(summary, default=str, indent=2))