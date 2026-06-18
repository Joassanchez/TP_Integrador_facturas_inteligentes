"""Integration tests for src/autoencoder/pipeline.py and comparison.py.

Tests validate: prerequisite validation (RNF-03), comparison CSV schema,
latent vector metadata, experimental artifact isolation, and the
non-modification contract (no classifier_baseline.keras).

Heavy TF training is mocked via ``unittest.mock.patch`` on Model.fit
and Model.predict to keep tests fast and deterministic.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.autoencoder.comparison import (
    build_classification_metrics,
    write_comparison_csv,
)
from src.autoencoder.pipeline import (
    _build_latent_vectors_df,
    validate_prerequisites,
)


# ===========================================================================
# 1 — validate_prerequisites: missing artifacts (RNF-03)
# ===========================================================================


class TestValidatePrerequisitesMissing:
    """Tests for fail-fast prerequisite validation."""

    def test_missing_preprocessor(self, tmp_path):
        """GIVEN preprocessor pickle is absent
        WHEN validate_prerequisites is called
        THEN raises FileNotFoundError mentioning 'preprocessor'."""
        le = tmp_path / "label_encoder_categoria.pkl"
        le.touch()
        bm = tmp_path / "modelo_supervisado.keras"
        bm.touch()
        tr = tmp_path / "training_report.json"
        tr.touch()

        pp = tmp_path / "preprocessor_supervisado.pkl"  # does NOT exist

        with pytest.raises(FileNotFoundError, match="preprocessor"):
            validate_prerequisites(
                preprocessor_path=str(pp),
                label_encoder_path=str(le),
                baseline_model_path=str(bm),
                training_report_path=str(tr),
            )

    def test_missing_label_encoder(self, tmp_path):
        """GIVEN label encoder pickle is absent
        WHEN validate_prerequisites is called
        THEN raises FileNotFoundError mentioning 'label encoder'."""
        pp = tmp_path / "preprocessor_supervisado.pkl"
        pp.touch()
        le = tmp_path / "label_encoder_categoria.pkl"  # does NOT exist
        bm = tmp_path / "modelo_supervisado.keras"
        bm.touch()
        tr = tmp_path / "training_report.json"
        tr.touch()

        with pytest.raises(FileNotFoundError, match="label encoder"):
            validate_prerequisites(
                preprocessor_path=str(pp),
                label_encoder_path=str(le),
                baseline_model_path=str(bm),
                training_report_path=str(tr),
            )

    def test_missing_baseline_model(self, tmp_path):
        """GIVEN baseline model is absent
        WHEN validate_prerequisites is called
        THEN raises FileNotFoundError mentioning 'baseline model'."""
        pp = tmp_path / "preprocessor_supervisado.pkl"
        pp.touch()
        le = tmp_path / "label_encoder_categoria.pkl"
        le.touch()
        bm = tmp_path / "modelo_supervisado.keras"  # does NOT exist
        tr = tmp_path / "training_report.json"
        tr.touch()

        with pytest.raises(FileNotFoundError, match="baseline model"):
            validate_prerequisites(
                preprocessor_path=str(pp),
                label_encoder_path=str(le),
                baseline_model_path=str(bm),
                training_report_path=str(tr),
            )

    def test_missing_training_report(self, tmp_path):
        """GIVEN training report is absent
        WHEN validate_prerequisites is called
        THEN raises FileNotFoundError mentioning 'training report'."""
        pp = tmp_path / "preprocessor_supervisado.pkl"
        pp.touch()
        le = tmp_path / "label_encoder_categoria.pkl"
        le.touch()
        bm = tmp_path / "modelo_supervisado.keras"
        bm.touch()
        tr = tmp_path / "training_report.json"  # does NOT exist

        with pytest.raises(FileNotFoundError, match="training report"):
            validate_prerequisites(
                preprocessor_path=str(pp),
                label_encoder_path=str(le),
                baseline_model_path=str(bm),
                training_report_path=str(tr),
            )

    def test_multiple_missing(self, tmp_path):
        """GIVEN multiple base artifacts are absent
        WHEN validate_prerequisites is called
        THEN ALL missing paths are listed in the error message."""
        pp = tmp_path / "preprocessor_supervisado.pkl"  # missing
        le = tmp_path / "label_encoder_categoria.pkl"  # missing
        bm = tmp_path / "modelo_supervisado.keras"
        bm.touch()
        tr = tmp_path / "training_report.json"
        tr.touch()

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_prerequisites(
                preprocessor_path=str(pp),
                label_encoder_path=str(le),
                baseline_model_path=str(bm),
                training_report_path=str(tr),
            )

        msg = str(exc_info.value)
        assert "preprocessor" in msg, (
            "Error must list missing preprocessor"
        )
        assert "label encoder" in msg, (
            "Error must list missing label encoder"
        )

    def test_all_present_does_not_raise(self, tmp_path):
        """GIVEN all base artifacts exist
        WHEN validate_prerequisites is called
        THEN does not raise any exception."""
        pp = tmp_path / "preprocessor_supervisado.pkl"
        pp.touch()
        le = tmp_path / "label_encoder_categoria.pkl"
        le.touch()
        bm = tmp_path / "modelo_supervisado.keras"
        bm.touch()
        tr = tmp_path / "training_report.json"
        tr.touch()

        # Should not raise
        validate_prerequisites(
            preprocessor_path=str(pp),
            label_encoder_path=str(le),
            baseline_model_path=str(bm),
            training_report_path=str(tr),
        )


# ===========================================================================
# 2 — comparison_metrics.csv: classification-only, 2 rows
# ===========================================================================


class TestComparisonCSV:
    """Tests for write_comparison_csv contract."""

    def test_comparison_csv_classification_only(self, tmp_path):
        """GIVEN two classification rows with metric columns
        WHEN written via write_comparison_csv
        THEN the CSV has exactly 2 data rows and no anomaly columns."""
        rows = [
            {
                "model": "baseline_supervisado",
                "accuracy": 0.85,
                "precision_macro": 0.80,
                "recall_macro": 0.78,
                "f1_macro": 0.79,
                "precision_weighted": 0.84,
                "recall_weighted": 0.85,
                "f1_weighted": 0.84,
            },
            {
                "model": "encoder_classifier",
                "accuracy": 0.72,
                "precision_macro": 0.68,
                "recall_macro": 0.66,
                "f1_macro": 0.67,
                "precision_weighted": 0.71,
                "recall_weighted": 0.72,
                "f1_weighted": 0.71,
            },
        ]

        out_path = tmp_path / "comparison_metrics.csv"
        write_comparison_csv(rows, str(out_path))

        df = pd.read_csv(out_path)
        assert len(df) == 2, (
            f"Expected exactly 2 rows, got {len(df)}"
        )
        assert "model" in df.columns
        assert "anomaly" not in " ".join(df.columns).lower(), (
            "Anomaly columns must not appear in comparison CSV"
        )

        # Verify both model names are present
        assert set(df["model"]) == {
            "baseline_supervisado",
            "encoder_classifier",
        }, f"Unexpected model names: {set(df['model'])}"


# ===========================================================================
# 3 — build_classification_metrics
# ===========================================================================


class TestBuildClassificationMetrics:
    """Tests for metric computation."""

    def test_perfect_classification(self):
        """GIVEN perfect predictions
        WHEN metrics are computed
        THEN all metrics equal 1.0."""
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

        metrics = build_classification_metrics(
            y_true, y_pred, class_names=["a", "b", "c"]
        )

        for key in metrics:
            assert metrics[key] == pytest.approx(1.0, abs=1e-6), (
                f"Expected {key}=1.0, got {metrics[key]}"
            )

    def test_imperfect_classification(self):
        """GIVEN imperfect predictions
        WHEN metrics are computed
        THEN accuracy is <1.0 and metrics are finite."""
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        y_pred = np.array([0, 1, 1, 0, 2, 2], dtype=np.int32)

        metrics = build_classification_metrics(
            y_true, y_pred, class_names=["a", "b", "c"]
        )

        assert metrics["accuracy"] < 1.0
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, (
                f"{key} must be in [0, 1], got {val}"
            )


# ===========================================================================
# 4 — latent_vectors.csv metadata
# ===========================================================================


class TestLatentVectorsCSV:
    """Tests for latent_vectors.csv schema."""

    def test_latent_vectors_csv_has_metadata(self):
        """GIVEN latent vectors from train and test splits
        WHEN _build_latent_vectors_df is called
        THEN the resulting DataFrame has id_factura, categoria, split columns
        plus latent_0..latent_n columns."""
        n_samples = 10
        latent_dim = 8

        id_train = np.array([f"F{i:03d}" for i in range(5)])
        cat_train = np.array(["a", "b", "a", "b", "a"])
        latent_train = np.random.RandomState(42).rand(5, latent_dim)

        id_test = np.array([f"F{i:03d}" for i in range(5, 10)])
        cat_test = np.array(["c", "c", "b", "a", "c"])
        latent_test = np.random.RandomState(42).rand(5, latent_dim)

        df = _build_latent_vectors_df(
            id_train=id_train,
            cat_train=cat_train,
            latent_train=latent_train,
            id_test=id_test,
            cat_test=cat_test,
            latent_test=latent_test,
        )

        # Metadata columns must be present
        for col in ("id_factura", "categoria", "split"):
            assert col in df.columns, (
                f"Missing metadata column: {col}"
            )

        # Latent columns must be present
        for i in range(latent_dim):
            assert f"latent_{i}" in df.columns, (
                f"Missing latent column: latent_{i}"
            )

        # Check row counts
        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
        assert (df["split"] == "train").sum() == 5
        assert (df["split"] == "test").sum() == 5

        # Metadata values preserved
        assert list(df.loc[df["split"] == "train", "id_factura"]) == list(
            id_train
        )

    def test_latent_vectors_shape_consistency(self):
        """GIVEN latent vectors with known dimension
        WHEN the DataFrame is built
        THEN latent column count matches the input dimension."""
        latent_dim = 5
        latent_train = np.zeros((3, latent_dim))
        latent_test = np.zeros((2, latent_dim))

        df = _build_latent_vectors_df(
            id_train=np.array(["a", "b", "c"]),
            cat_train=np.array(["x", "y", "z"]),
            latent_train=latent_train,
            id_test=np.array(["d", "e"]),
            cat_test=np.array(["w", "v"]),
            latent_test=latent_test,
        )

        latent_cols = [c for c in df.columns if c.startswith("latent_")]
        assert len(latent_cols) == latent_dim, (
            f"Expected {latent_dim} latent columns, got {len(latent_cols)}"
        )


# ===========================================================================
# 5 — Pipeline integration: experimental dirs + non-modification contract
# ===========================================================================


class TestPipelineIntegration:
    """Mocked integration tests for the full pipeline orchestration."""

    @pytest.fixture
    def base_artifacts(self, tmp_path):
        """Create minimal base artifacts for pipeline testing."""
        import json

        models_dir = tmp_path / "models"
        results_dir = tmp_path / "results"
        models_dir.mkdir()
        results_dir.mkdir()

        # Create training_report.json with baseline metrics
        report = {
            "metrics": {
                "accuracy": 0.85,
                "precision_macro": 0.80,
                "recall_macro": 0.78,
                "f1_macro": 0.79,
                "precision_weighted": 0.84,
                "recall_weighted": 0.85,
                "f1_weighted": 0.84,
            }
        }
        report_path = results_dir / "training_report.json"
        report_path.write_text(json.dumps(report))

        # Create dummy file placeholders (will be replaced by pickle in tests)
        pp_path = models_dir / "preprocessor_supervisado.pkl"
        pp_path.touch()
        le_path = models_dir / "label_encoder_categoria.pkl"
        le_path.touch()
        bm_path = models_dir / "modelo_supervisado.keras"
        bm_path.touch()

        return {
            "tmp_path": tmp_path,
            "models_dir": models_dir,
            "results_dir": results_dir,
            "pp_path": pp_path,
            "le_path": le_path,
            "bm_path": bm_path,
            "report_path": report_path,
        }

    def test_pipeline_saves_to_experimental_dirs(
        self, base_artifacts, tmp_path
    ):
        """GIVEN the pipeline runs (with mocked TF training)
        WHEN artifacts are written
        THEN models go to models/experimental/ and results go to
        results/experimental/."""
        from sklearn.preprocessing import LabelEncoder

        from src.autoencoder.pipeline import run_autoencoder_experiment

        ba = base_artifacts
        exp_models = ba["tmp_path"] / "models" / "experimental"
        exp_results = ba["tmp_path"] / "results" / "experimental"

        # Minimal dataset CSV that the pipeline can load
        data_path = ba["tmp_path"] / "data.csv"
        pd.DataFrame(
            {
                "id_factura": [f"F{i:03d}" for i in range(30)],
                "rubro": ["R1"] * 30,
                "subrubro": ["S1"] * 30,
                "categoria": ["cat_a"] * 10 + ["cat_b"] * 10 + ["cat_c"] * 10,
                "descripcion": [f"desc {i}" for i in range(30)],
                "fecha": pd.date_range("2024-01-01", periods=30),
            }
        ).to_csv(data_path, index=False)

        # Mock TF model methods to avoid real training
        mock_history = MagicMock()
        mock_history.history = {"loss": [0.5, 0.3], "val_loss": [0.4, 0.25]}

        # Build mock preprocessor and label encoder
        mock_pp = MagicMock()
        mock_pp.transform.side_effect = lambda X: (
            np.random.RandomState(42)
            .rand(len(X), 1052)
            .astype(np.float32)
        )

        mock_le = LabelEncoder()
        mock_le.fit(["cat_a", "cat_b", "cat_c"])

        # Mock Model.predict with call-counted side_effect so output
        # shapes match the expected model (autoencoder=same dim,
        # encoder=latent_dim, classifier=n_classes).
        _predict_call = [0]

        def _mock_predict(x, verbose=0):
            _predict_call[0] += 1
            n = x.shape[0]
            if _predict_call[0] <= 3:
                out_dim = x.shape[1]  # autoencoder: reconstruction
            elif _predict_call[0] <= 5:
                out_dim = 8  # encoder: latent_dim
            else:
                out_dim = 3  # classifier: n_classes
            return (
                np.random.RandomState(42).rand(n, out_dim).astype(np.float32)
            )

        with patch(
            "tensorflow.keras.Model.fit", return_value=mock_history
        ):
            with patch(
                "tensorflow.keras.Model.predict",
                side_effect=_mock_predict,
            ):
                def _mock_save(filepath, overwrite=True, **kwargs):
                    Path(filepath).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    Path(filepath).touch()

                with patch(
                    "tensorflow.keras.Model.save",
                    side_effect=_mock_save,
                ):
                    with patch(
                        "src.autoencoder.pipeline.derive_categoria",
                        side_effect=lambda df: df,
                    ):
                        with patch(
                            "src.autoencoder.pipeline.joblib.load",
                            side_effect=lambda p: (
                                mock_pp
                                if "preprocessor" in str(p)
                                else mock_le
                            ),
                        ):
                            # Build args namespace
                            class Args:
                                pass

                            args = Args()
                            args.input = str(data_path)
                            args.latent_dim = 8
                            args.threshold_k = 2.0
                            args.epochs = 2
                            args.patience = 5
                            args.preprocessor_path = str(ba["pp_path"])
                            args.label_encoder_path = str(ba["le_path"])
                            args.baseline_model_path = str(ba["bm_path"])
                            args.training_report_path = str(
                                ba["report_path"]
                            )
                            args.output_dir_models = str(exp_models)
                            args.output_dir_results = str(exp_results)
                            args.input_path = str(data_path)

                            summary = run_autoencoder_experiment(args)

        # Verify artifacts are in experimental directories
        exp_models_entries = list(exp_models.iterdir())
        exp_results_entries = list(exp_results.iterdir())

        model_names = {p.name for p in exp_models_entries}
        result_names = {p.name for p in exp_results_entries}

        assert "autoencoder.keras" in model_names, (
            f"Missing autoencoder.keras in {model_names}"
        )
        assert "encoder.keras" in model_names, (
            f"Missing encoder.keras in {model_names}"
        )
        assert "classifier_encoder.keras" in model_names, (
            f"Missing classifier_encoder.keras in {model_names}"
        )
        assert "autoencoder_report.json" in result_names, (
            f"Missing autoencoder_report.json in {result_names}"
        )
        assert "reconstruction_errors.csv" in result_names, (
            f"Missing reconstruction_errors.csv in {result_names}"
        )
        assert "latent_vectors.csv" in result_names, (
            f"Missing latent_vectors.csv in {result_names}"
        )
        assert "comparison_metrics.csv" in result_names, (
            f"Missing comparison_metrics.csv in {result_names}"
        )

        # Verify summary has expected keys
        assert "autoencoder" in summary
        assert "anomaly_detection" in summary
        assert "baseline_metrics" in summary
        assert "encoder_classifier_metrics" in summary

    def test_pipeline_does_not_create_classifier_baseline(
        self, base_artifacts, tmp_path
    ):
        """GIVEN the pipeline completes
        WHEN artifacts are written
        THEN classifier_baseline.keras is NOT created anywhere."""
        from sklearn.preprocessing import LabelEncoder

        from src.autoencoder.pipeline import run_autoencoder_experiment

        ba = base_artifacts
        exp_models = ba["tmp_path"] / "models" / "experimental"
        exp_results = ba["tmp_path"] / "results" / "experimental"

        # Minimal dataset
        data_path = ba["tmp_path"] / "data.csv"
        pd.DataFrame(
            {
                "id_factura": [f"F{i:03d}" for i in range(30)],
                "rubro": ["R1"] * 30,
                "subrubro": ["S1"] * 30,
                "categoria": ["cat_a"] * 10 + ["cat_b"] * 10 + ["cat_c"] * 10,
                "descripcion": [f"desc {i}" for i in range(30)],
                "fecha": pd.date_range("2024-01-01", periods=30),
            }
        ).to_csv(data_path, index=False)

        mock_history = MagicMock()
        mock_history.history = {"loss": [0.5], "val_loss": [0.4]}

        mock_pp = MagicMock()
        mock_pp.transform.side_effect = lambda X: (
            np.random.RandomState(42)
            .rand(len(X), 1052)
            .astype(np.float32)
        )

        mock_le = LabelEncoder()
        mock_le.fit(["cat_a", "cat_b", "cat_c"])

        _predict_call = [0]

        def _mock_predict(x, verbose=0):
            _predict_call[0] += 1
            n = x.shape[0]
            if _predict_call[0] <= 3:
                out_dim = x.shape[1]
            elif _predict_call[0] <= 5:
                out_dim = 8
            else:
                out_dim = 3
            return (
                np.random.RandomState(42).rand(n, out_dim).astype(np.float32)
            )

        with patch(
            "tensorflow.keras.Model.fit", return_value=mock_history
        ):
            with patch(
                "tensorflow.keras.Model.predict",
                side_effect=_mock_predict,
            ):
                def _mock_save2(filepath, overwrite=True, **kwargs):
                    Path(filepath).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    Path(filepath).touch()

                with patch(
                    "tensorflow.keras.Model.save",
                    side_effect=_mock_save2,
                ):
                    with patch(
                        "src.autoencoder.pipeline.derive_categoria",
                        side_effect=lambda df: df,
                    ):
                        with patch(
                            "src.autoencoder.pipeline.joblib.load",
                            side_effect=lambda p: (
                                mock_pp
                                if "preprocessor" in str(p)
                                else mock_le
                            ),
                        ):
                            class Args:
                                pass

                            args = Args()
                            args.input = str(data_path)
                            args.latent_dim = 8
                            args.threshold_k = 2.0
                            args.epochs = 2
                            args.patience = 5
                            args.preprocessor_path = str(ba["pp_path"])
                            args.label_encoder_path = str(ba["le_path"])
                            args.baseline_model_path = str(ba["bm_path"])
                            args.training_report_path = str(
                                ba["report_path"]
                            )
                            args.output_dir_models = str(exp_models)
                            args.output_dir_results = str(exp_results)
                            args.input_path = str(data_path)

                            run_autoencoder_experiment(args)

        # Verify classifier_baseline.keras is NOT present anywhere
        all_files = list(ba["tmp_path"].rglob("classifier_baseline*"))
        assert len(all_files) == 0, (
            f"classifier_baseline.keras must NOT be created, "
            f"but found: {all_files}"
        )
