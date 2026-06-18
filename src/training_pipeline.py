"""Training pipeline orchestrator: load artifacts → reconstruct split →
transform → train → evaluate → save model + report.

Entry point: python -m src.training_pipeline
"""

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.data_quality import derive_categoria
from src.model import build_mlp, compile_mlp, to_dense
from src.preprocessor import extract_target


# ===========================================================================
# Artifact loading — REQ-01
# ===========================================================================

def load_artifacts(preprocessor_path: str, encoder_path: str):
    """Load preprocessor and label encoder from disk with type checks."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import LabelEncoder

    pp_path = Path(preprocessor_path)
    le_path = Path(encoder_path)

    if not pp_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    if not le_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    preprocessor = joblib.load(pp_path)
    encoder = joblib.load(le_path)

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Expected ColumnTransformer, got {type(preprocessor).__name__}"
        )
    if not isinstance(encoder, LabelEncoder):
        raise TypeError(
            f"Expected LabelEncoder, got {type(encoder).__name__}"
        )

    return preprocessor, encoder


# ===========================================================================
# Split reconstruction — REQ-02, REQ-03
# ===========================================================================

def reconstruct_split(df, label_encoder):
    """Reconstruct the deterministic train/test split from preprocessing."""
    X, y_raw, _ = extract_target(df)
    y = label_encoder.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def create_val_split(X_train, y_train):
    """Create stratified validation split from transformed training data."""
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    return X_train_final, X_val, y_train_final, y_val


# ===========================================================================
# Post-hoc metrics — REQ-08
# ===========================================================================

def compute_posthoc_metrics(model, X_test_dense, y_test, label_encoder) -> dict:
    """Compute precision, recall, and F1 (macro + weighted) on test set."""
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred_probs = model.predict(X_test_dense, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)

    return {
        "precision_macro": float(precision_score(
            y_test, y_pred, average="macro", zero_division=0,
        )),
        "recall_macro": float(recall_score(
            y_test, y_pred, average="macro", zero_division=0,
        )),
        "f1_macro": float(f1_score(
            y_test, y_pred, average="macro", zero_division=0,
        )),
        "precision_weighted": float(precision_score(
            y_test, y_pred, average="weighted", zero_division=0,
        )),
        "recall_weighted": float(recall_score(
            y_test, y_pred, average="weighted", zero_division=0,
        )),
        "f1_weighted": float(f1_score(
            y_test, y_pred, average="weighted", zero_division=0,
        )),
    }


# ===========================================================================
# Training orchestrator — REQ-01 through REQ-10
# ===========================================================================

def run_training(
    input_path: str = "data/raw/dataset_facturas_sistemas_inteligentes.xlsx",
    model_dir: str = "models",
    results_dir: str = "results",
) -> dict:
    """Run the full supervised training pipeline end-to-end."""
    # ---- Reproducibility seeds ----
    tf.random.set_seed(42)
    np.random.seed(42)

    model_p = Path(model_dir)
    results_p = Path(results_dir)
    model_p.mkdir(parents=True, exist_ok=True)
    results_p.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load dataset ----
    df, _metadata = load_dataset(input_path)

    # ---- 2. Derive canonical categoria ----
    df = derive_categoria(df)

    # ---- 3. Load artifacts ----
    preprocessor_path = model_p / "preprocessor_supervisado.pkl"
    encoder_path = model_p / "label_encoder_categoria.pkl"
    preprocessor, label_encoder = load_artifacts(
        str(preprocessor_path), str(encoder_path),
    )

    # ---- 4. Reconstruct split ----
    X_train, X_test, y_train, y_test = reconstruct_split(df, label_encoder)

    # ---- 5. Transform with fitted preprocessor ----
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # ---- 6. Create validation split ----
    X_train_final, X_val, y_train_final, y_val = create_val_split(
        X_train_proc, y_train,
    )

    # ---- 7. Convert to dense ----
    X_train_dense = to_dense(X_train_final)
    X_val_dense = to_dense(X_val)
    X_test_dense = to_dense(X_test_proc)

    # ---- 8. Build and compile ----
    n_features = X_train_dense.shape[1]
    n_classes = len(label_encoder.classes_)
    model = build_mlp(n_features, n_classes)
    compile_mlp(model)

    # ---- 9. Train ----
    keras_path = str(model_p / "modelo_supervisado.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=keras_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train_dense, y_train_final,
        validation_data=(X_val_dense, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- 10. Evaluate on test ----
    test_loss, test_accuracy = model.evaluate(
        X_test_dense, y_test, verbose=0,
    )

    # ---- 11. Save model ----
    model.save(keras_path)

    # ---- 12. Post-hoc metrics ----
    posthoc = compute_posthoc_metrics(
        model, X_test_dense, y_test, label_encoder,
    )

    # ---- 13. Build class distribution ----
    y_train_raw = label_encoder.inverse_transform(y_train_final)
    y_val_raw = label_encoder.inverse_transform(y_val)
    y_test_raw = label_encoder.inverse_transform(y_test)

    train_counts = dict(Counter(y_train_raw))
    val_counts = dict(Counter(y_val_raw))
    test_counts = dict(Counter(y_test_raw))

    all_classes = sorted(
        set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())
    )
    class_dist = {}
    for cls in all_classes:
        class_dist[cls] = {
            "train_final": train_counts.get(cls, 0),
            "val": val_counts.get(cls, 0),
            "test": test_counts.get(cls, 0),
        }

    # ---- 14. Build report ----
    stopped_epoch = len(history.history.get("loss", []))

    report = {
        "architecture": {
            "input_dim": n_features,
            "output_dim": n_classes,
            "layers": [
                "Input",
                "Dense(256,relu)", "Dropout(0.3)",
                "Dense(128,relu)", "Dropout(0.3)",
                "Dense(64,relu)",
                "Dense(n_classes,softmax)",
            ],
        },
        "hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": 32,
            "epochs_max": 200,
        },
        "training_config": {
            "early_stopping_patience": 10,
            "early_stopping_monitor": "val_loss",
            "validation_strategy": "explicit_stratified_split",
            "seed": 42,
        },
        "metrics": {
            "accuracy": float(test_accuracy),
            **posthoc,
        },
        "history": {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        },
        "artifact_paths": {
            "model": keras_path,
            "report": str(results_p / "training_report.json"),
            "preprocessor": str(preprocessor_path),
            "label_encoder": str(encoder_path),
        },
        "shapes": {
            "X_train": list(X_train_dense.shape),
            "X_val": list(X_val_dense.shape),
            "X_test": list(X_test_dense.shape),
        },
        "class_distribution": class_dist,
        "stopped_epoch": stopped_epoch,
    }

    # ---- 15. Write report ----
    report_path = results_p / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return report


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Supervised training pipeline"
    )
    parser.add_argument(
        "--input",
        default="data/raw/dataset_facturas_sistemas_inteligentes.xlsx",
        help="Path to input XLSX or CSV dataset",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory for serialized artifacts",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for training_report.json",
    )

    args = parser.parse_args()

    report = run_training(
        input_path=args.input,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
