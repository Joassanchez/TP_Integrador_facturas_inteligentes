"""Autoencoder experiment pipeline: prerequisite checks, autoencoder training,
anomaly detection, latent-vector classification, and comparison against the
supervised baseline.

Entry point: python -m src.autoencoder.pipeline

All new artifacts live under models/experimental/ and results/experimental/.
Existing src/data/, src/preprocessing/, and src/training/ are read-only.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.autoencoder.model import (
    anomaly_flag,
    build_autoencoder,
    build_encoder,
    reconstruction_error,
)
from src.autoencoder.comparison import (
    build_classification_metrics,
    write_comparison_csv,
)
from src.autoencoder.plots import (
    plot_confusion_matrix,
    plot_reconstruction_error_distribution,
    plot_training_loss,
)
from src.data.loader import load_dataset
from src.data.quality import derive_categoria
from src.preprocessing.preprocessor import extract_target
from src.training.model import compile_mlp, to_dense


# ===========================================================================
# Prerequisite validation — RNF-03
# ===========================================================================


def validate_prerequisites(
    preprocessor_path: str = "models/preprocessor_supervisado.pkl",
    label_encoder_path: str = "models/label_encoder_categoria.pkl",
    baseline_model_path: str = "models/modelo_supervisado.keras",
    training_report_path: str = "results/training_report.json",
) -> None:
    """Fail early with a clear message listing ALL missing artifacts.

    RNF-03: The pipeline MUST NOT regenerate base artifacts. If any are
    missing, raise FileNotFoundError with every missing path listed.

    Args:
        preprocessor_path: Path to the fitted ColumnTransformer pickle.
        label_encoder_path: Path to the fitted LabelEncoder pickle.
        baseline_model_path: Path to the supervised baseline .keras model.
        training_report_path: Path to the supervised training_report.json.

    Raises:
        FileNotFoundError: If one or more required artifacts are missing,
            listing ALL missing paths in the error message.
    """
    required = {
        "preprocessor": Path(preprocessor_path),
        "label encoder": Path(label_encoder_path),
        "baseline model": Path(baseline_model_path),
        "training report": Path(training_report_path),
    }

    missing = [
        f"  - {name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        msg = (
            "Missing required base artifacts — run the supervised pipeline "
            "first:\n"
            "  python -m src.preprocessing.pipeline\n"
            "  python -m src.training.pipeline\n\n"
            "The following files were not found:\n"
            + "\n".join(missing)
        )
        raise FileNotFoundError(msg)


# ===========================================================================
# Helper: build classifier for encoder latent vectors
# ===========================================================================


def _build_classifier(n_features: int, n_classes: int) -> tf.keras.Sequential:
    """Build a small MLP classifier for latent-vector classification.

    Architecture is intentionally smaller than the supervised baseline
    because the input dimension is the latent_dim (e.g. 24), not 1052.

    Args:
        n_features: Number of latent dimensions (input features).
        n_classes: Number of output classes from the LabelEncoder.

    Returns:
        A compiled tf.keras.Sequential classifier.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="classifier_encoder",
    )

    compile_mlp(model, learning_rate=0.001)
    return model


# ===========================================================================
# Orchestrator
# ===========================================================================


def run_autoencoder_experiment(args) -> dict:
    """Run the full autoencoder experiment end-to-end.

    Orchestrates prerequisite validation, data loading, autoencoder training,
    anomaly detection, latent-vector classification, and comparison against
    the supervised baseline. All experimental artifacts are written under
    ``models/experimental/`` and ``results/experimental/``.

    Args:
        args: argparse.Namespace with all CLI parameters.

    Returns:
        dict: Experiment summary with metrics, paths, and counts.
    """
    # ---- 1. Validate prerequisites (fail-fast) ----
    validate_prerequisites(
        preprocessor_path=args.preprocessor_path,
        label_encoder_path=args.label_encoder_path,
        baseline_model_path=args.baseline_model_path,
        training_report_path=args.training_report_path,
    )

    # ---- 2. Reproducibility seeds ----
    tf.random.set_seed(42)
    np.random.seed(42)

    # ---- 3. Headless plotting ----
    matplotlib.use("Agg")

    # ---- 4. Load preprocessor + label encoder ----
    preprocessor = joblib.load(args.preprocessor_path)
    label_encoder = joblib.load(args.label_encoder_path)
    n_classes = len(label_encoder.classes_)

    # ---- 5. Load & prepare data ----
    df, _metadata = load_dataset(args.input_path)
    df = derive_categoria(df)

    # Preserve id_factura before extract_target drops it
    id_factura_series = df["id_factura"].copy()

    # Extract features and raw target (discard the new LabelEncoder)
    X_features, y_raw, _new_le = extract_target(df)
    y = label_encoder.transform(y_raw)

    # ---- 6. Reconstruct deterministic split ----
    n_samples = len(X_features)
    all_indices = np.arange(n_samples)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        idx_train,
        idx_test,
    ) = train_test_split(
        X_features,
        y,
        all_indices,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Retrieve metadata for each split
    id_train = id_factura_series.iloc[idx_train].values
    id_test = id_factura_series.iloc[idx_test].values
    cat_train = y_raw.iloc[idx_train].values
    cat_test = y_raw.iloc[idx_test].values

    # ---- 7. Transform via preprocessor → dynamic n_features ----
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    n_features = X_train_proc.shape[1]  # NEVER hardcoded

    # ---- 8. Convert sparse to dense ----
    X_train_dense = to_dense(X_train_proc)
    X_test_dense = to_dense(X_test_proc)

    # ---- 9. Train autoencoder ----
    autoencoder = build_autoencoder(
        n_features=n_features, latent_dim=args.latent_dim
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    history = autoencoder.fit(
        X_train_dense,
        X_train_dense,  # autoencoder target = input
        epochs=args.epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- 10. Save models ----
    models_dir = Path(args.output_dir_models)
    models_dir.mkdir(parents=True, exist_ok=True)

    autoencoder_path = models_dir / "autoencoder.keras"
    encoder_path = models_dir / "encoder.keras"

    autoencoder.save(str(autoencoder_path))

    encoder = build_encoder(autoencoder, latent_dim=args.latent_dim)
    encoder.save(str(encoder_path))

    # ---- 11. Compute reconstruction errors on ALL data ----
    results_dir = Path(args.output_dir_results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Combine train+test for global error distribution
    X_all_dense = np.vstack([X_train_dense, X_test_dense])
    X_all_hat = autoencoder.predict(X_all_dense, verbose=0)
    errors_all = reconstruction_error(X_all_dense, X_all_hat)

    # Separate errors per split (for per-row CSV)
    X_train_hat = autoencoder.predict(X_train_dense, verbose=0)
    X_test_hat = autoencoder.predict(X_test_dense, verbose=0)
    errors_train = reconstruction_error(X_train_dense, X_train_hat)
    errors_test = reconstruction_error(X_test_dense, X_test_hat)

    # ---- 12. Derive threshold ----
    threshold_mean = float(np.mean(errors_all))
    threshold_std = float(np.std(errors_all))
    threshold_value = threshold_mean + args.threshold_k * threshold_std

    # ---- 13. Flag anomalies ----
    flags_train = anomaly_flag(errors_train, threshold_value)
    flags_test = anomaly_flag(errors_test, threshold_value)

    # ---- 14. Save autoencoder_report.json ----
    n_normal = int((~np.concatenate([flags_train, flags_test])).sum())
    n_atypical = int(np.concatenate([flags_train, flags_test]).sum())

    training_history = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }

    report = {
        "architecture": {
            "type": "autoencoder",
            "input_dim": n_features,
            "latent_dim": args.latent_dim,
            "hidden": [256, 128],
            "loss": "mse",
            "optimizer": "Adam(learning_rate=1e-3)",
        },
        "hyperparameters": {
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": 32,
            "validation_split": 0.2,
        },
        "training_history": training_history,
        "stopped_epoch": len(history.history.get("loss", [])),
        "anomaly_detection": {
            "threshold_method": "mean_plus_k_std",
            "threshold_k": args.threshold_k,
            "threshold_value": threshold_value,
            "threshold_mean": threshold_mean,
            "threshold_std": threshold_std,
            "n_normal": n_normal,
            "n_atypical": n_atypical,
            "mean_reconstruction_error": float(np.mean(errors_all)),
            "max_reconstruction_error": float(np.max(errors_all)),
        },
        "artifact_paths": {
            "autoencoder": str(autoencoder_path),
            "encoder": str(encoder_path),
            "report": str(results_dir / "autoencoder_report.json"),
            "reconstruction_errors_csv": str(
                results_dir / "reconstruction_errors.csv"
            ),
            "latent_vectors_csv": str(results_dir / "latent_vectors.csv"),
        },
    }

    report_path = results_dir / "autoencoder_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ---- 15. Save reconstruction_errors.csv ----
    rec_errors_df = _build_reconstruction_errors_df(
        id_train=id_train,
        cat_train=cat_train,
        errors_train=errors_train,
        flags_train=flags_train,
        id_test=id_test,
        cat_test=cat_test,
        errors_test=errors_test,
        flags_test=flags_test,
        threshold=threshold_value,
    )
    rec_errors_path = results_dir / "reconstruction_errors.csv"
    rec_errors_df.to_csv(rec_errors_path, index=False)

    # ---- 16. Extract latent vectors ----
    latent_train = encoder.predict(X_train_dense, verbose=0)
    latent_test = encoder.predict(X_test_dense, verbose=0)

    latent_df = _build_latent_vectors_df(
        id_train=id_train,
        cat_train=cat_train,
        latent_train=latent_train,
        id_test=id_test,
        cat_test=cat_test,
        latent_test=latent_test,
    )
    latent_path = results_dir / "latent_vectors.csv"
    latent_df.to_csv(latent_path, index=False)

    # ---- 17. Load baseline metrics (no retrain) ----
    baseline_report = json.loads(
        Path(args.training_report_path).read_text(encoding="utf-8")
    )
    baseline_metrics = {
        "accuracy": float(baseline_report["metrics"]["accuracy"]),
        "precision_macro": float(
            baseline_report["metrics"]["precision_macro"]
        ),
        "recall_macro": float(baseline_report["metrics"]["recall_macro"]),
        "f1_macro": float(baseline_report["metrics"]["f1_macro"]),
        "precision_weighted": float(
            baseline_report["metrics"]["precision_weighted"]
        ),
        "recall_weighted": float(
            baseline_report["metrics"]["recall_weighted"]
        ),
        "f1_weighted": float(baseline_report["metrics"]["f1_weighted"]),
    }

    # ---- 18. Train classifier_encoder on latent vectors ----
    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]

    # Training split only
    latent_train_subset = latent_df.loc[
        latent_df["split"] == "train", latent_cols
    ].values.astype(np.float32)

    classifier = _build_classifier(
        n_features=args.latent_dim, n_classes=n_classes
    )

    clf_history = classifier.fit(
        latent_train_subset,
        y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )

    # ---- 19. Save classifier_encoder ----
    classifier_path = models_dir / "classifier_encoder.keras"
    classifier.save(str(classifier_path))

    # ---- 20. Compute classification metrics for classifier_encoder ----
    # Predict on the corresponding test set (same y_test as baseline)
    latent_test_subset = latent_df.loc[
        latent_df["split"] == "test", latent_cols
    ].values.astype(np.float32)

    y_pred_probs = classifier.predict(latent_test_subset, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)

    class_names = list(label_encoder.classes_)
    encoder_metrics = build_classification_metrics(
        y_test, y_pred, class_names
    )

    # ---- 21. Write comparison_metrics.csv (classification-only, 2 rows) ----
    comparison_rows = [
        {"model": "baseline_supervisado", **baseline_metrics},
        {"model": "encoder_classifier", **encoder_metrics},
    ]
    comparison_path = results_dir / "comparison_metrics.csv"
    write_comparison_csv(comparison_rows, str(comparison_path))

    # ---- 22. Generate plots ----
    plot_training_loss(
        history,
        str(results_dir / "autoencoder_training_loss.png"),
    )
    plot_reconstruction_error_distribution(
        errors_all,
        threshold_value,
        str(results_dir / "autoencoder_error_distribution.png"),
    )
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        "Classifier on Encoder Latent Vectors — Confusion Matrix",
        str(results_dir / "encoder_classifier_confusion.png"),
    )

    # ---- 23. Build return dict ----
    return {
        "autoencoder": {
            "input_dim": n_features,
            "latent_dim": args.latent_dim,
            "stopped_epoch": report["stopped_epoch"],
        },
        "anomaly_detection": report["anomaly_detection"],
        "baseline_metrics": baseline_metrics,
        "encoder_classifier_metrics": encoder_metrics,
        "artifact_paths": {
            "autoencoder": str(autoencoder_path),
            "encoder": str(encoder_path),
            "classifier_encoder": str(classifier_path),
            "autoencoder_report": str(report_path),
            "reconstruction_errors_csv": str(rec_errors_path),
            "latent_vectors_csv": str(latent_path),
            "comparison_metrics_csv": str(comparison_path),
        },
    }


# ===========================================================================
# CSV builders
# ===========================================================================


def _build_reconstruction_errors_df(
    id_train,
    cat_train,
    errors_train,
    flags_train,
    id_test,
    cat_test,
    errors_test,
    flags_test,
    threshold,
) -> pd.DataFrame:
    """Build the per-row reconstruction_errors.csv DataFrame."""
    train_part = pd.DataFrame(
        {
            "id_factura": id_train,
            "categoria": cat_train,
            "split": "train",
            "reconstruction_error": errors_train,
            "threshold": threshold,
            "anomaly_label": flags_train,
        }
    )
    test_part = pd.DataFrame(
        {
            "id_factura": id_test,
            "categoria": cat_test,
            "split": "test",
            "reconstruction_error": errors_test,
            "threshold": threshold,
            "anomaly_label": flags_test,
        }
    )
    return pd.concat([train_part, test_part], ignore_index=True)


def _build_latent_vectors_df(
    id_train,
    cat_train,
    latent_train,
    id_test,
    cat_test,
    latent_test,
) -> pd.DataFrame:
    """Build the latent_vectors.csv DataFrame with metadata columns."""
    latent_dim = latent_train.shape[1]
    latent_cols = [f"latent_{i}" for i in range(latent_dim)]

    train_part = pd.DataFrame(
        {
            "id_factura": id_train,
            "categoria": cat_train,
            "split": "train",
        }
    )
    train_latent = pd.DataFrame(latent_train, columns=latent_cols)
    train_combined = pd.concat([train_part, train_latent], axis=1)

    test_part = pd.DataFrame(
        {
            "id_factura": id_test,
            "categoria": cat_test,
            "split": "test",
        }
    )
    test_latent = pd.DataFrame(latent_test, columns=latent_cols)
    test_combined = pd.concat([test_part, test_latent], axis=1)

    return pd.concat([train_combined, test_combined], ignore_index=True)


# ===========================================================================
# CLI entry point
# ===========================================================================


def main():
    """CLI entry point: python -m src.autoencoder.pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experimental autoencoder pipeline"
    )
    parser.add_argument(
        "--input",
        default="data/raw/dataset_facturas_sistemas_inteligentes.xlsx",
        help="Path to input XLSX or CSV dataset",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=24,
        help="Bottleneck dimension (default: 24)",
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=2.0,
        help="Anomaly threshold: mean + k*std (default: 2.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="EarlyStopping patience (default: 10)",
    )
    parser.add_argument(
        "--preprocessor-path",
        default="models/preprocessor_supervisado.pkl",
        help="Path to fitted preprocessor pickle",
    )
    parser.add_argument(
        "--label-encoder-path",
        default="models/label_encoder_categoria.pkl",
        help="Path to fitted label encoder pickle",
    )
    parser.add_argument(
        "--baseline-model-path",
        default="models/modelo_supervisado.keras",
        help="Path to baseline supervised model",
    )
    parser.add_argument(
        "--training-report-path",
        default="results/training_report.json",
        help="Path to baseline training report JSON",
    )
    parser.add_argument(
        "--output-dir-models",
        default="models/experimental",
        help="Directory for experimental model artifacts",
    )
    parser.add_argument(
        "--output-dir-results",
        default="results/experimental",
        help="Directory for experimental result artifacts",
    )

    args = parser.parse_args()

    # Auto-create output directories
    Path(args.output_dir_models).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir_results).mkdir(parents=True, exist_ok=True)

    summary = run_autoencoder_experiment(args)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("  Autoencoder Experiment — Summary")
    print("=" * 60)
    print(
        f"  Input dim      : {summary['autoencoder']['input_dim']}"
    )
    print(
        f"  Latent dim     : {summary['autoencoder']['latent_dim']}"
    )
    print(
        f"  Epochs trained : {summary['autoencoder']['stopped_epoch']}"
    )
    print()
    print("  Anomaly Detection")
    print(
        f"    Method       : {summary['anomaly_detection']['threshold_method']}"
    )
    print(
        f"    Threshold    : {summary['anomaly_detection']['threshold_value']:.6f}"
    )
    print(
        f"    Normal       : {summary['anomaly_detection']['n_normal']}"
    )
    print(
        f"    Atypical     : {summary['anomaly_detection']['n_atypical']}"
    )
    print()
    print("  Classification Comparison")
    bl = summary["baseline_metrics"]
    ec = summary["encoder_classifier_metrics"]
    print(f"    Baseline (supervised)      : acc={bl['accuracy']:.4f}")
    print(f"    Encoder classifier (latent): acc={ec['accuracy']:.4f}")
    print()
    print("  Artifacts")
    for name, path in summary["artifact_paths"].items():
        print(f"    {name}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
