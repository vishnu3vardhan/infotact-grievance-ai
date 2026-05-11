"""Final evaluation script for department and sentiment models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_macro_f1,
    calculate_precision,
    calculate_recall,
    generate_classification_report,
    plot_confusion_matrix,
)
from src.models.predict import DepartmentPredictor
from src.models.predict_sentiment import SentimentPredictor
from src.models.train_sentiment_model import SentimentModelTrainer, SentimentTrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "text" not in df.columns and "processed_text" not in df.columns:
        raise ValueError("Dataset must contain either 'text' or 'processed_text'.")
    return df


def evaluate_department_model(
    dataset_path: str | Path = "data/interim/cleaned_data.csv",
    model_path: str | Path = "artifacts/models/best_department_model.pkl",
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate the department model on a held-out split."""
    df = _load_dataset(dataset_path)
    if "department" not in df.columns:
        raise ValueError("Dataset must contain the 'department' column.")

    text_column = "text" if "text" in df.columns else "processed_text"
    df = df[[text_column, "department"]].dropna().copy()
    df[text_column] = df[text_column].astype(str)
    df["department"] = df["department"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column],
        df["department"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["department"],
    )

    predictor = DepartmentPredictor(model_path=model_path)
    predictions = predictor.predict(list(X_test))

    metrics = {
        "accuracy": calculate_accuracy(y_test, predictions),
        "precision_macro": calculate_precision(y_test, predictions),
        "recall_macro": calculate_recall(y_test, predictions),
        "f1_macro": calculate_macro_f1(y_test, predictions),
        "classification_report": generate_classification_report(y_test, predictions),
    }

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        labels=sorted(df["department"].unique()),
        title="Department Model Confusion Matrix",
        save_path="reports/department_confusion_matrix.png",
    )
    return metrics


def evaluate_sentiment_model(
    dataset_path: str | Path = "data/interim/cleaned_data.csv",
    model_path: str | Path = "artifacts/models/best_sentiment_model.pkl",
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate the sentiment model on a held-out split."""
    trainer = SentimentModelTrainer(config=SentimentTrainingConfig(dataset_path=Path(dataset_path)))
    df = trainer._balance_minority_classes(trainer._load_and_validate_dataset())
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["sentiment"],
    )

    predictor = SentimentPredictor(model_path=model_path)
    predictions = predictor.predict(list(X_test))
    prediction_labels = [prediction.sentiment for prediction in predictions]

    metrics = {
        "accuracy": calculate_accuracy(y_test, prediction_labels),
        "precision_macro": calculate_precision(y_test, prediction_labels),
        "recall_macro": calculate_recall(y_test, prediction_labels),
        "f1_macro": calculate_macro_f1(y_test, prediction_labels),
        "classification_report": generate_classification_report(y_test, prediction_labels),
    }

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=prediction_labels,
        labels=list(predictor.label_encoder.classes_),
        title="Sentiment Model Confusion Matrix",
        save_path="reports/sentiment_confusion_matrix.png",
    )
    return metrics


def main() -> dict[str, Any]:
    """Run final evaluation for department and sentiment models."""
    department_metrics = evaluate_department_model()
    sentiment_metrics = evaluate_sentiment_model()

    summary = {
        "department": department_metrics,
        "sentiment": sentiment_metrics,
    }

    logger.info("Department metrics: %s", json.dumps({k: v for k, v in department_metrics.items() if k != 'classification_report'}, indent=2))
    logger.info("Sentiment metrics: %s", json.dumps({k: v for k, v in sentiment_metrics.items() if k != 'classification_report'}, indent=2))
    return summary


if __name__ == "__main__":
    main()
