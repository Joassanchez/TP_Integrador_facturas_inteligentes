"""Unit tests for src/autoencoder/model.py.

Tests validate: autoencoder construction (Functional API), encoder extraction,
reconstruction error, anomaly detection boundary, and a real smoke training run.

All TF-dependent tests are skipped when TensorFlow is not installed via
pytest.importorskip at module level.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from src.autoencoder.model import (
    anomaly_flag,
    build_autoencoder,
    build_encoder,
    reconstruction_error,
)


# ===========================================================================
# 1 — build_autoencoder: shapes, compile state, dynamic input
# ===========================================================================


class TestBuildAutoencoder:
    """Tests for Functional API autoencoder construction."""

    @pytest.fixture(autouse=True)
    def seed_everything(self):
        """Set TF and numpy seeds before each test for determinism."""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_build_autoencoder_returns_model(self):
        """GIVEN n_features=100
        WHEN build_autoencoder is called
        THEN returns compiled tf.keras.Model with correct input/output shapes,
        MSE loss, and Adam optimizer."""
        model = build_autoencoder(n_features=100, latent_dim=24)

        assert isinstance(model, tf.keras.Model), (
            f"Expected tf.keras.Model, got {type(model)}"
        )
        assert model.input_shape == (None, 100), (
            f"Expected input_shape (None, 100), got {model.input_shape}"
        )
        assert model.output_shape == (None, 100), (
            f"Expected output_shape (None, 100), got {model.output_shape}"
        )

        # Verify compiled state
        assert model.loss == "mse", (
            f"Expected MSE loss, got {model.loss}"
        )
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam), (
            f"Expected Adam optimizer, got {type(model.optimizer)}"
        )

        # Verify functional API (not Sequential)
        assert not isinstance(model, tf.keras.Sequential), (
            "Expected Functional API model, not Sequential"
        )

    @pytest.mark.parametrize("n_features", [50, 100, 1052])
    def test_build_autoencoder_dynamic_input(self, n_features):
        """GIVEN different n_features values
        WHEN build_autoencoder is called
        THEN model input/output shapes adapt to the dynamic value."""
        model = build_autoencoder(n_features=n_features, latent_dim=24)

        assert model.input_shape == (None, n_features), (
            f"Expected input_shape (None, {n_features}), "
            f"got {model.input_shape}"
        )
        assert model.output_shape == (None, n_features), (
            f"Expected output_shape (None, {n_features}), "
            f"got {model.output_shape}"
        )

        # Bottleneck layer must have latent_dim=24 units
        latent_layer = model.get_layer(name="bottleneck")
        assert latent_layer.units == 24, (
            f"Expected bottleneck dim 24, got {latent_layer.units}"
        )

    def test_build_autoencoder_architecture_structure(self):
        """GIVEN an autoencoder built with defaults
        WHEN inspecting layers
        THEN encoder/decoder are symmetric: 256→128→latent→128→256→n_features,
        output activation is linear."""
        model = build_autoencoder(n_features=100, latent_dim=24)

        # Collect Dense layers in order
        dense_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dense)
        ]

        assert len(dense_layers) == 6, (
            f"Expected 6 Dense layers, got {len(dense_layers)}"
        )

        # Encoder: 256, 128, latent=24
        assert dense_layers[0].units == 256, (
            f"Expected enc_dense_1=256, got {dense_layers[0].units}"
        )
        assert dense_layers[0].activation.__name__ == "relu"

        assert dense_layers[1].units == 128, (
            f"Expected enc_dense_2=128, got {dense_layers[1].units}"
        )
        assert dense_layers[1].activation.__name__ == "relu"

        assert dense_layers[2].units == 24, (
            f"Expected bottleneck=24, got {dense_layers[2].units}"
        )
        assert dense_layers[2].activation.__name__ == "relu"

        # Decoder: 128, 256, n_features
        assert dense_layers[3].units == 128, (
            f"Expected dec_dense_1=128, got {dense_layers[3].units}"
        )
        assert dense_layers[3].activation.__name__ == "relu"

        assert dense_layers[4].units == 256, (
            f"Expected dec_dense_2=256, got {dense_layers[4].units}"
        )
        assert dense_layers[4].activation.__name__ == "relu"

        # Output layer: n_features, linear
        assert dense_layers[5].units == 100, (
            f"Expected output=100, got {dense_layers[5].units}"
        )
        assert dense_layers[5].activation.__name__ == "linear"


# ===========================================================================
# 2 — build_encoder: sub-graph extraction
# ===========================================================================


class TestBuildEncoder:
    """Tests for encoder sub-model extraction."""

    @pytest.fixture(autouse=True)
    def seed_everything(self):
        """Set TF and numpy seeds before each test for determinism."""
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_build_encoder_returns_subgraph(self):
        """GIVEN a built autoencoder
        WHEN build_encoder extracts the encoder
        THEN returns a Model mapping (B, n_features) → (B, latent_dim),
        sharing the autoencoder's input layer."""
        autoencoder = build_autoencoder(n_features=80, latent_dim=16)
        encoder = build_encoder(autoencoder, latent_dim=16)

        assert isinstance(encoder, tf.keras.Model), (
            f"Expected tf.keras.Model, got {type(encoder)}"
        )
        assert encoder.input_shape == (None, 80), (
            f"Expected input_shape (None, 80), got {encoder.input_shape}"
        )
        assert encoder.output_shape == (None, 16), (
            f"Expected output_shape (None, 16), got {encoder.output_shape}"
        )

        # Encoder input must be the same as autoencoder input
        assert encoder.input is autoencoder.input, (
            "Encoder must share the autoencoder's input layer"
        )

    def test_build_encoder_produces_valid_latent(self):
        """GIVEN a built autoencoder and extracted encoder
        WHEN encoder is called on synthetic data
        THEN output has correct shape and values are finite."""
        autoencoder = build_autoencoder(n_features=50, latent_dim=8)
        encoder = build_encoder(autoencoder, latent_dim=8)

        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 50)).astype(np.float32)
        latent = encoder.predict(X, verbose=0)

        assert latent.shape == (10, 8), (
            f"Expected latent shape (10, 8), got {latent.shape}"
        )
        assert np.all(np.isfinite(latent)), "Latent values must be finite"

    def test_build_encoder_latent_dim_validation(self):
        """GIVEN an autoencoder with latent_dim=10
        WHEN build_encoder is called with latent_dim=20
        THEN raises ValueError."""
        autoencoder = build_autoencoder(n_features=50, latent_dim=10)

        with pytest.raises(ValueError, match="does not match expected"):
            build_encoder(autoencoder, latent_dim=20)


# ===========================================================================
# 3 — reconstruction_error: shape, non-negative, perfect reconstruction
# ===========================================================================


class TestReconstructionError:
    """Tests for per-row MSE reconstruction error."""

    def test_reconstruction_error_shape_and_nonnegative(self):
        """GIVEN X shape (15, 200) and X_hat same shape
        WHEN reconstruction_error is called
        THEN returns (15,) non-negative array."""
        rng = np.random.default_rng(42)
        X = rng.random((15, 200)).astype(np.float32)
        X_hat = X + rng.normal(0, 0.1, size=(15, 200)).astype(np.float32)

        errors = reconstruction_error(X, X_hat)

        assert errors.shape == (15,), (
            f"Expected shape (15,), got {errors.shape}"
        )
        assert np.all(errors >= 0), "All errors must be non-negative"

    def test_reconstruction_error_perfect(self):
        """GIVEN X equals X_hat exactly
        WHEN reconstruction_error is called
        THEN all error values are zero."""
        rng = np.random.default_rng(123)
        X = rng.random((20, 50)).astype(np.float32)
        X_hat = X.copy()

        errors = reconstruction_error(X, X_hat)

        assert errors.shape == (20,)
        np.testing.assert_allclose(errors, np.zeros(20), atol=1e-7)

    def test_reconstruction_error_shape_mismatch(self):
        """GIVEN X and X_hat with mismatched shapes
        WHEN reconstruction_error is called
        THEN raises ValueError."""
        X = np.zeros((10, 100), dtype=np.float32)
        X_hat = np.zeros((10, 99), dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            reconstruction_error(X, X_hat)


# ===========================================================================
# 4 — anomaly_flag: boundary, percentile
# ===========================================================================


class TestAnomalyFlag:
    """Tests for threshold-based anomaly flagging."""

    def test_anomaly_flag_boundary(self):
        """GIVEN errors [0.1, 0.5, 1.0] and threshold=0.5
        WHEN anomaly_flag is called
        THEN values strictly above threshold are True, at/below are False."""
        errors = np.array([0.1, 0.5, 1.0, 0.5, 0.5001], dtype=np.float32)
        threshold = 0.5

        flags = anomaly_flag(errors, threshold)

        assert flags.dtype == bool
        expected = np.array([False, False, True, False, True])
        np.testing.assert_array_equal(flags, expected)

    def test_anomaly_flag_percentile(self):
        """GIVEN 1000 reconstruction errors
        WHEN flagged with threshold at 95th percentile
        THEN approximately 5% are flagged (45-55 due to sampling)."""
        rng = np.random.default_rng(42)

        # Generate right-skewed errors (typical for reconstruction)
        normal_errors = rng.normal(0.1, 0.05, size=950)
        anomaly_errors = rng.normal(0.5, 0.2, size=50)
        errors = np.concatenate([normal_errors, anomaly_errors])

        threshold = np.quantile(errors, 0.95)
        flags = anomaly_flag(errors, threshold)

        n_flagged = np.sum(flags)
        # Should be approximately 5% of 1000 = 50, allow tolerance
        assert 30 <= n_flagged <= 70, (
            f"Expected ~50 flagged, got {n_flagged}"
        )


# ===========================================================================
# 5 — Smoke test: real training on synthetic data
# ===========================================================================


class TestAutoencoderSmoke:
    """Real training smoke test on synthetic data (1 epoch)."""

    def test_autoencoder_smoke_train(self):
        """GIVEN synthetic data (30, 50) with a known structure
        WHEN autoencoder is built and trained for 1 epoch
        THEN loss is finite, reconstruction has correct shape,
        and reconstruction errors are non-negative."""
        tf.random.set_seed(42)
        np.random.seed(42)

        n_features = 50
        n_samples = 30

        # Create structured synthetic data: sine waves with noise
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2 * np.pi, n_features)
        X = np.zeros((n_samples, n_features), dtype=np.float32)
        for i in range(n_samples):
            phase = rng.uniform(0, np.pi)
            X[i] = np.sin(t + phase) + rng.normal(0, 0.05, size=n_features)

        # Build autoencoder
        model = build_autoencoder(n_features=n_features, latent_dim=8)

        # Train 1 epoch
        history = model.fit(
            X, X,
            epochs=1,
            batch_size=8,
            verbose=0,
        )

        # Loss must be finite
        final_loss = history.history["loss"][-1]
        assert np.isfinite(final_loss), (
            f"Training loss must be finite, got {final_loss}"
        )

        # Reconstruct
        X_hat = model.predict(X, verbose=0)

        # Output shape must match input
        assert X_hat.shape == (n_samples, n_features), (
            f"Expected reconstruction shape {(n_samples, n_features)}, "
            f"got {X_hat.shape}"
        )

        # Reconstruction errors must be non-negative
        errors = reconstruction_error(X, X_hat)
        assert np.all(errors >= 0), "Reconstruction errors must be non-negative"
        assert errors.shape == (n_samples,), (
            f"Expected errors shape ({n_samples},), got {errors.shape}"
        )
