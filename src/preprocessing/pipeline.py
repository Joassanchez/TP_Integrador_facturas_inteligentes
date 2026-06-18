"""Preprocessing pipeline orchestrator: load → derive target → leak-drop →
split → fit-on-train → transform → serialize.

Entry point: python -m src.preprocessing.pipeline
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.data.quality import derive_categoria
from src.preprocessing.preprocessor import (
    build_preprocessor,
    extract_target,
    validate_class_counts,
)


def run_preprocessing(
    input_path: str = "data/raw/dataset_facturas_sistemas_inteligentes.xlsx",
    model_dir: str = "models",
    results_dir: str = "results",
) -> dict:
    """Run the full supervised preprocessing pipeline end-to-end.

    1. Load dataset from input_path.
    2. Derive canonical categoria.
    3. Separate features (X) and target (y_raw) with leak-column drop.
    4. Validate every class has ≥2 samples.
    5. Encode target via LabelEncoder.
    6. Stratified train/test split (80/20, random_state=42).
    7. Fit ColumnTransformer on X_train ONLY.
    8. Transform X_train and X_test.
    9. Serialize preprocessor + label_encoder.
    10. Write preprocessing_report.json.
    11. Return report dict.

    Args:
        input_path: Path to XLSX or CSV dataset.
        model_dir: Directory for .pkl artifacts.
        results_dir: Directory for preprocessing_report.json.

    Returns:
        dict: The preprocessing report.
    """
    # ---- 1. Load ----
    df, _metadata = load_dataset(input_path)

    # ---- 2. Derive canonical categoria ----
    df = derive_categoria(df)

    # ---- 3. Separate features and target ----
    X, y_raw, label_encoder = extract_target(df)

    # ---- 4. Class-count guard ----
    validate_class_counts(y_raw)

    # ---- 5. Encode target ----
    y = label_encoder.transform(y_raw)

    # ---- 6. Stratified split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---- 7. Build and fit preprocessor on X_train ONLY ----
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # ---- 8. Transform ----
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # ---- 9. Serialize artifacts ----
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    preprocessor_path = model_path / "preprocessor_supervisado.pkl"
    encoder_path = model_path / "label_encoder_categoria.pkl"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, encoder_path)

    # ---- 10. Build report ----
    # Class distribution across train/test
    y_train_raw = label_encoder.inverse_transform(y_train)
    y_test_raw = label_encoder.inverse_transform(y_test)

    train_counts = pd_value_counts(y_train_raw)
    test_counts = pd_value_counts(y_test_raw)

    all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    class_distribution = {}
    for cls in all_classes:
        class_distribution[cls] = {
            "train": train_counts.get(cls, 0),
            "test": test_counts.get(cls, 0),
        }

    # Detect matrix type (sparse vs dense)
    if hasattr(X_train_proc, "toarray"):
        matrix_type = type(X_train_proc).__name__
        output_type = "scipy sparse matrix"
        # csr_matrix stores non-zero values in .data; NaN can only appear there
        train_nan = bool(np.isnan(X_train_proc.data).any()) if hasattr(X_train_proc, "data") else False
        test_nan = bool(np.isnan(X_test_proc.data).any()) if hasattr(X_test_proc, "data") else False
    else:
        matrix_type = "numpy ndarray"
        output_type = "numpy ndarray"
        train_nan = bool(np.isnan(X_train_proc).any())
        test_nan = bool(np.isnan(X_test_proc).any())

    report = {
        "output_type": output_type,
        "matrix_type": matrix_type,
        "input_shape": list(X.shape),
        "train_shape": list(X_train_proc.shape),
        "test_shape": list(X_test_proc.shape),
        "feature_count": X_train_proc.shape[1],
        "nan_in_train": bool(train_nan),
        "nan_in_test": bool(test_nan),
        "class_distribution": class_distribution,
        "artifact_paths": {
            "preprocessor": str(preprocessor_path),
            "label_encoder": str(encoder_path),
        },
        "random_state": 42,
    }

    # ---- 11. Write report JSON ----
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    report_file = results_path / "preprocessing_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pd_value_counts(series) -> dict:
    """Safe value_counts returning a plain dict."""
    import pandas as pd

    if isinstance(series, pd.Series):
        counts = series.value_counts()
        return {str(k): int(v) for k, v in counts.items()}
    # numpy array
    unique, counts = np.unique(series, return_counts=True)
    return {str(k): int(v) for k, v in zip(unique, counts)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Supervised preprocessing pipeline"
    )
    parser.add_argument(
        "--input",
        default="data/raw/dataset_facturas_sistemas_inteligentes.xlsx",
        help="Path to input XLSX or CSV dataset",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory for serialized .pkl artifacts",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for preprocessing_report.json",
    )

    args = parser.parse_args()

    report = run_preprocessing(
        input_path=args.input,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
