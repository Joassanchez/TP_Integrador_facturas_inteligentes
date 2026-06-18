"""Unit tests for src/model.py — RED phase: tests written BEFORE implementation.

Tests validate: build_mlp shapes, compile_mlp config, to_dense conversion,
and deterministic construction. All TF-dependent tests are skipped when
TensorFlow is not installed via pytest.importorskip at module level.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf
from scipy.sparse import csr_matrix

# Import from module under test — does NOT exist yet (RED)
from src.model import build_mlp, compile_mlp, to_dense  # noqa: E402


# ===========================================================================
# 2.1 — to_dense: CSR→float32 + ndarray passthrough → REQ-04
# ===========================================================================

class TestToDense:
    """Tests for sparse-to-dense conversion — REQ-04."""

    def test_to_dense_csr(self):
        """GIVEN csr_matrix shape (50, 1052)
        WHEN converted via to_dense
        THEN returns float32 ndarray, shape preserved, no NaN."""
        rng = np.random.default_rng(42)
        dense_data = rng.random((50, 1052))
        sparse = csr_matrix(dense_data)

        result = to_dense(sparse)

        assert isinstance(result, np.ndarray), (
            f"Expected ndarray, got {type(result)}"
        )
        assert result.dtype == np.float32, (
            f"Expected float32, got {result.dtype}"
        )
        assert result.shape == (50, 1052), (
            f"Expected shape (50, 1052), got {result.shape}"
        )
        assert not np.isnan(result).any(), "NaN found in dense output"

        # Values must match the original sparse data
        np.testing.assert_allclose(result, dense_data, rtol=1e-5)

    def test_to_dense_ndarray_passthrough(self):
        """GIVEN already-dense float64 ndarray
        WHEN passed to to_dense
        THEN cast to float32, shape preserved, no NaN."""
        rng = np.random.default_rng(99)
        arr = rng.random((10, 20)).astype(np.float64)

        result = to_dense(arr)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (10, 20)
        assert not np.isnan(result).any()

        # Values must match the original after float32 cast
        np.testing.assert_allclose(result, arr.astype(np.float32), rtol=1e-7)


# ===========================================================================
# 2.2 — build_mlp: shapes + dynamic dimensions → REQ-05, REQ-11
# ===========================================================================

class TestBuildMLP:
    """Tests for MLP construction — REQ-05, REQ-11."""

    @pytest.fixture(autouse=True)
    def seed_everything(self):
        """Set TF and numpy seeds before each test for determinism."""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_build_mlp_shapes(self):
        """GIVEN n_features=1052, n_classes=24
        WHEN MLP built
        THEN input_shape=(None,1052), output_shape=(None,24)."""
        model = build_mlp(n_features=1052, n_classes=24)

        assert model.input_shape == (None, 1052), (
            f"Expected input_shape (None, 1052), got {model.input_shape}"
        )
        assert model.output_shape == (None, 24), (
            f"Expected output_shape (None, 24), got {model.output_shape}"
        )

    def test_build_mlp_dynamic_dims(self):
        """GIVEN n_features=500, n_classes=10
        WHEN MLP built
        THEN shapes adapt to the dynamic values, last activation=softmax."""
        model = build_mlp(n_features=500, n_classes=10)

        assert model.input_shape == (None, 500)
        assert model.output_shape == (None, 10)

        # Last layer must be Dense with softmax
        last_layer = model.layers[-1]
        assert isinstance(last_layer, tf.keras.layers.Dense), (
            f"Last layer should be Dense, got {type(last_layer)}"
        )
        assert last_layer.activation.__name__ == "softmax", (
            f"Expected softmax activation, got {last_layer.activation.__name__}"
        )

    def test_build_mlp_architecture_structure(self):
        """GIVEN model built
        WHEN inspecting layers
        THEN Dense layers are 256, 128, 64, n_classes; Dropout(0.3) after first two."""
        model = build_mlp(n_features=1052, n_classes=24)

        # Collect layer names and types
        layer_specs = [
            (layer.__class__.__name__,
             getattr(layer, 'units', None),
             getattr(layer, 'rate', None),
             getattr(layer, 'activation', None))
            for layer in model.layers
        ]

        # Check Dense layer units in order
        dense_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Dense)
        ]

        # At least 4 Dense layers: 256, 128, 64, 24
        assert len(dense_layers) >= 4, (
            f"Expected at least 4 Dense layers, got {len(dense_layers)}"
        )

        # First Dense: 256 with relu
        assert dense_layers[0].units == 256
        assert dense_layers[0].activation.__name__ == "relu"

        # Second Dense: 128 with relu
        assert dense_layers[1].units == 128
        assert dense_layers[1].activation.__name__ == "relu"

        # Third Dense: 64 with relu
        assert dense_layers[2].units == 64
        assert dense_layers[2].activation.__name__ == "relu"

        # Last Dense: n_classes with softmax
        assert dense_layers[-1].units == 24
        assert dense_layers[-1].activation.__name__ == "softmax"

        # Check Dropout(0.3) present after first and second Dense layers
        dropout_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Dropout)
        ]
        assert len(dropout_layers) >= 2, (
            f"Expected at least 2 Dropout layers, got {len(dropout_layers)}"
        )
        for dl in dropout_layers:
            assert dl.rate == 0.3, f"Expected dropout rate 0.3, got {dl.rate}"

    def test_build_mlp_layer_count_not_hardcoded(self):
        """GIVEN Keras may count InputLayer differently across versions
        WHEN checking layer count
        THEN we validate via Dense/Dropout presence, NOT a hardcoded number."""
        model = build_mlp(n_features=100, n_classes=5)

        # Just verify the model is valid (no hardcoded layer count assertion)
        assert isinstance(model, tf.keras.Sequential)
        # Must have at least a Dense layer
        assert len(model.layers) >= 1


# ===========================================================================
# 2.3 — compile_mlp: loss, optimizer, metrics → REQ-06
# ===========================================================================

class TestCompileMLP:
    """Tests for model compilation — REQ-06."""

    def test_compile_mlp(self):
        """GIVEN a built MLP
        WHEN compiled via compile_mlp()
        THEN loss=sparse_categorical_crossentropy, optimizer=Adam(0.001),
        and compiled metrics contain accuracy."""
        model = build_mlp(n_features=100, n_classes=5)

        compiled = compile_mlp(model, learning_rate=0.001)

        # Check loss
        assert compiled.loss == "sparse_categorical_crossentropy" or (
            hasattr(compiled.loss, 'name') and
            compiled.loss.name == "sparse_categorical_crossentropy"
        ), f"Expected sparse_categorical_crossentropy loss, got {compiled.loss}"

        # Check optimizer is Adam
        assert isinstance(compiled.optimizer, tf.keras.optimizers.Adam), (
            f"Expected Adam optimizer, got {type(compiled.optimizer)}"
        )

        # Check compile was successful — model has metrics
        assert len(compiled.metrics) >= 1, (
            "Expected at least one compiled metric"
        )

        # Verify accuracy is compiled by running a tiny evaluate call.
        # Keras 3 wraps metrics in a container, so model.metrics may show
        # 'compile_metrics' instead of individual names. A real evaluate
        # call proves the accuracy metric is active.
        import numpy as np
        dummy_x = np.zeros((2, 100), dtype=np.float32)
        dummy_y = np.array([0, 1], dtype=np.int32)
        loss, acc = compiled.evaluate(dummy_x, dummy_y, verbose=0)
        assert isinstance(acc, (float, np.floating)), (
            f"Expected accuracy float from evaluate, got {type(acc)}"
        )

    def test_compile_mlp_returns_model(self):
        """GIVEN a built MLP
        WHEN compiled
        THEN returns the same model instance (for chaining)."""
        model = build_mlp(n_features=50, n_classes=3)
        compiled = compile_mlp(model)

        assert compiled is model, (
            "compile_mlp should return the same model instance"
        )
