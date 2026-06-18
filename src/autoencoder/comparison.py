"""Classification-only comparison utilities for the autoencoder experiment.

Exposes build_classification_metrics (accuracy, precision, recall, F1 —
macro and weighted) and write_comparison_csv (2-row classification
comparison against the supervised baseline).
"""

import csv
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def build_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> dict:
    """Compute multiclass classification metrics for comparison.

    Returns a dict with accuracy, and precision/recall/F1 (macro + weighted).
    All metrics use ``zero_division=0`` to handle edge cases gracefully.

    Args:
        y_true: Ground-truth integer labels, shape (n_samples,).
        y_pred: Predicted integer labels, shape (n_samples,).
        class_names: List of class name strings (for provenance only;
            not used in metric computation but retained for traceability).

    Returns:
        dict with keys: accuracy, precision_macro, recall_macro, f1_macro,
        precision_weighted, recall_weighted, f1_weighted.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
    }


def write_comparison_csv(rows: list[dict], path: str) -> None:
    """Write a classification-only comparison CSV.

    The output CSV contains exactly the rows provided — typically two rows:
    ``baseline_supervisado`` and ``encoder_classifier``. Columns include
    ``model`` plus all metric keys present in the first row. Anomaly data
    is NEVER included; this is classification-only per the resolved design.

    Args:
        rows: List of dicts, each with a ``model`` key and metric columns.
        path: Output file path. Parent directories are created if needed.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("rows must not be empty")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Use fieldnames from first row to preserve insertion order
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
