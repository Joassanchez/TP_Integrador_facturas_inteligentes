"""Autoencoder model construction: Functional API, encoder extraction, and
anomaly detection helpers.

Exposes build_autoencoder, build_encoder, reconstruction_error, and
anomaly_flag. Follows tf-keras-training skill: dynamic shapes, Adam
optimizer, dense float32, and Functional API.
"""

import numpy as np
import tensorflow as tf


def build_autoencoder(
    n_features: int,
    latent_dim: int = 24,
    hidden: tuple = (256, 128),
) -> tf.keras.Model:
    """Build a symmetric autoencoder using the Functional API.

    Architecture (symmetric):
        Input(n_features) → Dense(hidden[0], relu) → Dense(hidden[1], relu)
        → Dense(latent_dim, relu) [bottleneck]
        → Dense(hidden[1], relu) → Dense(hidden[0], relu)
        → Dense(n_features, linear)

    The model is compiled with MSE loss and Adam(learning_rate=1e-3).

    Args:
        n_features: Number of input features (dynamic, from preprocessor output).
        latent_dim: Bottleneck dimension (default 24).
        hidden: Tuple of (encoder_hidden_1, encoder_hidden_2) units.
            Defaults to (256, 128). Decoder mirrors this in reverse.

    Returns:
        A compiled tf.keras.Model.

    Raises:
        ValueError: If hidden is not a 2-element tuple.
    """
    if len(hidden) != 2:
        raise ValueError(
            f"hidden must be a 2-element tuple, got {hidden}"
        )

    h1, h2 = hidden

    # ---- Encoder ----
    inputs = tf.keras.layers.Input(shape=(n_features,), name="input")
    x = tf.keras.layers.Dense(h1, activation="relu", name="enc_dense_1")(inputs)
    x = tf.keras.layers.Dense(h2, activation="relu", name="enc_dense_2")(x)
    latent = tf.keras.layers.Dense(
        latent_dim, activation="relu", name="bottleneck"
    )(x)

    # ---- Decoder ----
    x = tf.keras.layers.Dense(h2, activation="relu", name="dec_dense_1")(latent)
    x = tf.keras.layers.Dense(h1, activation="relu", name="dec_dense_2")(x)
    outputs = tf.keras.layers.Dense(
        n_features, activation="linear", name="output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    return model


def build_encoder(
    autoencoder: tf.keras.Model, latent_dim: int
) -> tf.keras.Model:
    """Extract the encoder as an independent sub-model from the autoencoder.

    Slices the autoencoder's functional graph at the bottleneck layer.
    The returned model maps (B, n_features) → (B, latent_dim).

    Args:
        autoencoder: A trained autoencoder built by build_autoencoder().
        latent_dim: The bottleneck dimension (used only for validation).

    Returns:
        A tf.keras.Model representing the encoder sub-graph.

    Raises:
        ValueError: If the bottleneck layer is not found.
    """
    try:
        latent_layer = autoencoder.get_layer(name="bottleneck")
    except ValueError:
        raise ValueError(
            "bottleneck layer not found in autoencoder. "
            "Ensure the autoencoder was built with build_autoencoder()."
        )

    # Validate output dimension matches expected latent_dim
    if latent_layer.units != latent_dim:
            raise ValueError(
                f"Bottleneck output dimension {latent_layer.units} "
                f"does not match expected latent_dim={latent_dim}"
            )

    encoder = tf.keras.Model(
        inputs=autoencoder.input,
        outputs=latent_layer.output,
        name="encoder",
    )

    return encoder


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """Compute per-row Mean Squared Error between original and reconstructed.

    Args:
        X: Original features, shape (n_samples, n_features).
        X_hat: Reconstructed features, same shape as X.

    Returns:
        1D array of shape (n_samples,) with non-negative MSE values.

    Raises:
        ValueError: If shapes do not match.
    """
    if X.shape != X_hat.shape:
        raise ValueError(
            f"Shape mismatch: X has shape {X.shape}, "
            f"X_hat has shape {X_hat.shape}"
        )

    errors = np.mean(np.square(X - X_hat), axis=1)

    return errors


def anomaly_flag(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Flag anomalies based on a reconstruction error threshold.

    Args:
        errors: Reconstruction errors of shape (n_samples,).
        threshold: Values strictly above this are flagged as anomalies (True).

    Returns:
        Boolean array of shape (n_samples,), True where errors > threshold.
    """
    return errors > threshold
