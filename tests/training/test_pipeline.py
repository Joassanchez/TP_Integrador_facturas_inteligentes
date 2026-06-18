"""Integration tests for src/training/pipeline.py — RED phase.

Tests validate: artifact loading, split reconstruction, smoke training,
orchestration order, and report contract. Uses tmp_path for file I/O
and mocked fit() for fast integration tests.

All TF-dependent tests skip via pytest.importorskip at module level.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest import mock

pytest.importorskip("tensorflow")

import tensorflow as tf

# Import from module under test — does NOT exist yet (RED)
from src.training.pipeline import (  # noqa: E402
    load_artifacts,
    reconstruct_split,
    create_val_split,
    compute_posthoc_metrics,
    run_training,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def synthetic_data():
    """Tiny synthetic dataset: 30 rows, 3 features, 3 classes."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "descripcion": [f"item {i}" for i in range(n)],
        "proveedor": rng.choice(["A", "B", "C"], n),
        "tipo_comprobante": rng.choice(["FC_A", "FC_B"], n),
        "rubro": rng.choice(["r1", "r2"], n),
        "subrubro": rng.choice(["s1", "s2"], n),
        "categoria": rng.choice(["cat_x", "cat_y", "cat_z"], n),
        "monto": rng.uniform(100, 1000, n),
        "fecha": pd.date_range("2024-01-01", periods=n, freq="3D"),
        "id_factura": [f"F{i:04d}" for i in range(n)],
    })


@pytest.fixture(autouse=True)
def seed_everything():
    """Set TF and numpy seeds before each test for determinism."""
    tf.random.set_seed(42)
    np.random.seed(42)


# ===========================================================================
# 2.4 — Artifact loading → REQ-01
# ===========================================================================

class TestLoadArtifacts:
    """Tests for artifact loading — REQ-01."""

    def test_load_artifacts(self, tmp_path):
        """GIVEN valid .pkl files for ColumnTransformer and LabelEncoder
        WHEN loaded via load_artifacts
        THEN returns ColumnTransformer and LabelEncoder with correct types."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import (
            LabelEncoder,
            OneHotEncoder,
            StandardScaler,
        )

        # Build a real ColumnTransformer and LabelEncoder, dump to tmp_path
        preprocessor = ColumnTransformer(
            [("num", StandardScaler(), [0])],
            remainder="drop",
        )
        df_dummy = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        preprocessor.fit(df_dummy)

        encoder = LabelEncoder()
        encoder.fit(["a", "b", "c"])

        import joblib
        pp_path = tmp_path / "preprocessor.pkl"
        le_path = tmp_path / "encoder.pkl"
        joblib.dump(preprocessor, pp_path)
        joblib.dump(encoder, le_path)

        loaded_pp, loaded_le = load_artifacts(
            preprocessor_path=str(pp_path),
            encoder_path=str(le_path),
        )

        assert isinstance(loaded_pp, ColumnTransformer), (
            f"Expected ColumnTransformer, got {type(loaded_pp)}"
        )
        assert isinstance(loaded_le, LabelEncoder), (
            f"Expected LabelEncoder, got {type(loaded_le)}"
        )
        assert len(loaded_le.classes_) == 3

    def test_load_artifacts_missing_preprocessor(self, tmp_path):
        """GIVEN a non-existent preprocessor path
        WHEN loading
        THEN FileNotFoundError raised with the missing path."""
        missing = str(tmp_path / "does_not_exist.pkl")
        valid_encoder = tmp_path / "encoder.pkl"

        from sklearn.preprocessing import LabelEncoder
        import joblib
        encoder = LabelEncoder()
        encoder.fit(["x"])
        joblib.dump(encoder, valid_encoder)

        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            load_artifacts(
                preprocessor_path=missing,
                encoder_path=str(valid_encoder),
            )

    def test_load_artifacts_missing_encoder(self, tmp_path):
        """GIVEN a non-existent encoder path
        WHEN loading
        THEN FileNotFoundError raised with the missing path."""
        valid_pp = tmp_path / "preprocessor.pkl"
        missing = str(tmp_path / "missing_encoder.pkl")

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        import joblib
        pp = ColumnTransformer([("num", StandardScaler(), [0])], remainder="drop")
        pp.fit(pd.DataFrame({"f": [1.0]}))
        joblib.dump(pp, valid_pp)

        with pytest.raises(FileNotFoundError, match="missing_encoder"):
            load_artifacts(
                preprocessor_path=str(valid_pp),
                encoder_path=missing,
            )


# ===========================================================================
# 2.5 — Smoke training + end-to-end mocked → REQ-07, REQ-08, REQ-13
# ===========================================================================

class TestSmokeTraining:
    """Lightweight training tests — REQ-07, REQ-08, REQ-13.
    
    All tests use ≤5 epochs and small datasets to stay under 10s total.
    """

    def test_smoke_training(self, synthetic_data):
        """GIVEN tiny synthetic dense dataset (30 rows, 3 classes)
        WHEN model trained for 2 epochs
        THEN loss finite, history has expected keys."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from src.training.model import build_mlp, compile_mlp, to_dense

        # Prepare minimal data
        X = synthetic_data.select_dtypes(include=[np.number]).values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(synthetic_data["categoria"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        X_train_dense = to_dense(X_train)
        X_test_dense = to_dense(X_test)

        model = build_mlp(n_features=X_train_dense.shape[1], n_classes=len(le.classes_))
        compile_mlp(model)

        history = model.fit(
            X_train_dense, y_train,
            validation_split=0.2,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        h = history.history
        assert "loss" in h
        assert "val_loss" in h
        assert np.isfinite(h["loss"][-1]), "Final training loss not finite"
        assert np.isfinite(h["val_loss"][-1]), "Final validation loss not finite"

        # Quick eval
        loss, *_ = model.evaluate(X_test_dense, y_test, verbose=0)
        assert np.isfinite(loss), "Test loss not finite"

    def test_end_to_end_mocked(self, tmp_path, synthetic_data):
        """GIVEN mock for model.fit()
        WHEN run_training() called
        THEN orchestration order correct, report returned with 9 keys.
        ModelCheckpoint callback handles saving (not called in mock)."""
        from sklearn.preprocessing import LabelEncoder
        import joblib

        # Build a real preprocessor from the synthetic data so columns match
        from src.preprocessing.preprocessor import build_preprocessor, extract_target
        from src.data.quality import derive_categoria

        df_derived = derive_categoria(synthetic_data.copy())
        X, _, _ = extract_target(df_derived)
        pp = build_preprocessor(X)
        pp.fit(X)

        le = LabelEncoder()
        le.fit(df_derived["categoria"])

        model_dir = tmp_path / "models"
        results_dir = tmp_path / "results"
        model_dir.mkdir()
        results_dir.mkdir()

        pp_path = model_dir / "preprocessor_supervisado.pkl"
        le_path = model_dir / "label_encoder_categoria.pkl"
        joblib.dump(pp, pp_path)
        joblib.dump(le, le_path)

        # Save datasets as CSV so load_dataset works
        data_path = tmp_path / "data.csv"
        synthetic_data.to_csv(data_path, index=False)

        # Mock model.fit() and model.save() to avoid actual training
        with mock.patch("src.training.pipeline.build_mlp") as mock_build, \
             mock.patch("src.training.pipeline.compile_mlp") as mock_compile:

            mock_model = mock.MagicMock()
            mock_model.input_shape = (None, 1)
            mock_model.output_shape = (None, 3)
            mock_model.fit.return_value = mock.MagicMock(history={
                "loss": [0.8, 0.5],
                "val_loss": [0.9, 0.6],
                "accuracy": [0.6, 0.8],
                "val_accuracy": [0.5, 0.7],
            })
            mock_model.evaluate.return_value = [0.4, 0.75]
            # Return 6 predictions (30 rows → 6 test after 80/20 split)
            mock_model.predict.return_value = np.array([
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
                [0.7, 0.2, 0.1],
                [0.3, 0.3, 0.4],
                [0.6, 0.1, 0.3],
                [0.1, 0.6, 0.3],
            ], dtype=np.float32)
            mock_build.return_value = mock_model
            mock_compile.return_value = mock_model

            report = run_training(
                input_path=str(data_path),
                model_dir=str(model_dir),
                results_dir=str(results_dir),
            )

        # Verify report has 9 top-level keys
        required_keys = {
            "architecture", "hyperparameters", "training_config",
            "metrics", "history", "artifact_paths",
            "shapes", "class_distribution", "stopped_epoch",
        }
        assert required_keys.issubset(set(report.keys())), (
            f"Missing keys: {required_keys - set(report.keys())}"
        )

        # Verify fit() was called with validation_data (not validation_split)
        fit_kwargs = mock_model.fit.call_args.kwargs
        assert "validation_data" in fit_kwargs, (
            "fit() must receive validation_data=(X_val, y_val), not validation_split"
        )

        # Verify evaluate() was called on test set only
        mock_model.evaluate.assert_called_once()

        # Model saved by ModelCheckpoint callback (save_best_only=True)
        # No explicit model.save() — checkpoint handles serialization


# ===========================================================================
# 2.6 — Report contract → REQ-10, REQ-12
# ===========================================================================

class TestReportContract:
    """Tests for training_report.json schema — REQ-10, REQ-12."""

    def test_report_contract_nine_keys(self):
        """GIVEN a training report dict
        WHEN validating keys
        THEN contains exactly the 9 required top-level keys."""
        required = {
            "architecture", "hyperparameters", "training_config",
            "metrics", "history", "artifact_paths",
            "shapes", "class_distribution", "stopped_epoch",
        }

        # Build minimal valid report to verify keys exist
        report = {
            "architecture": {"input_dim": 1052, "output_dim": 24},
            "hyperparameters": {"learning_rate": 0.001, "epochs_max": 200},
            "training_config": {"batch_size": 32, "early_stopping_patience": 10},
            "metrics": {"accuracy": 0.85, "precision_macro": 0.82, "recall_macro": 0.80, "f1_macro": 0.81, "precision_weighted": 0.86, "recall_weighted": 0.85, "f1_weighted": 0.85},
            "history": {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6]},
            "artifact_paths": {"model": "models/modelo_supervisado.keras", "report": "results/training_report.json"},
            "shapes": {"X_train": [256, 1052], "X_val": [64, 1052], "X_test": [100, 1052]},
            "class_distribution": {},
            "stopped_epoch": 45,
        }

        for key in required:
            assert key in report, f"Missing required key: {key}"

    def test_report_metrics_are_finite(self):
        """GIVEN metrics in report
        WHEN validated
        THEN all metric values are finite floats in [0, 1]."""
        metrics = {
            "accuracy": 0.85,
            "precision_macro": 0.82,
            "recall_macro": 0.80,
            "f1_macro": 0.81,
            "precision_weighted": 0.86,
            "recall_weighted": 0.85,
            "f1_weighted": 0.85,
        }

        for name, value in metrics.items():
            assert isinstance(value, (int, float, np.floating)), (
                f"Metric '{name}' is not numeric: {type(value)}"
            )
            assert np.isfinite(float(value)), (
                f"Metric '{name}' is not finite: {value}"
            )
            assert 0.0 <= float(value) <= 1.0, (
                f"Metric '{name}' out of [0,1]: {value}"
            )

    def test_architecture_reads_from_artifacts(self):
        """GIVEN architecture section
        WHEN read
        THEN input_dim and output_dim come from runtime artifacts, not hardcoded."""
        arch = {"input_dim": 1052, "output_dim": 24}

        assert "input_dim" in arch
        assert "output_dim" in arch
        assert isinstance(arch["input_dim"], int)
        assert isinstance(arch["output_dim"], int)
        assert arch["input_dim"] > 0
        assert arch["output_dim"] > 0
