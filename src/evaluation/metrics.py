"""Reusable evaluation utilities for multiclass text classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calculate_accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Calculate accuracy score."""
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Calculate macro precision for multiclass output."""
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def calculate_recall(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Calculate macro recall for multiclass output."""
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def calculate_macro_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Calculate macro F1 score for multiclass output."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def generate_classification_report(
    y_true: Iterable[str], y_pred: Iterable[str]
) -> str:
    """Generate a text classification report."""
    return classification_report(y_true, y_pred, zero_division=0)


def plot_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: list[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | Path | None = None,
) -> None:
    """Plot multiclass confusion matrix and optionally save figure."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", output)

    plt.close()
