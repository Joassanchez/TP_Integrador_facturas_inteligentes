"""Model construction: MLP builder, compiler, and sparse-to-dense conversion.

Exposes build_mlp, compile_mlp, and to_dense for the supervised training
pipeline. Follows tf-keras-training skill: dynamic shapes, sparse_categorical
crossentropy, Adam optimizer, and dense float32 requirement for Keras layers.
"""

import numpy as np
import tensorflow as tf


def to_dense(X) -> np.ndarray:
    """Convert sparse matrix or ndarray to dense float32.

    CSR/CSC/COO matrices are converted via .toarray(). Already-dense
    ndarrays are cast to float32. Raises ValueError if NaN is found
    after conversion.

    Args:
        X: scipy sparse matrix or numpy ndarray.

    Returns:
        np.ndarray of dtype float32, same shape as input.

    Raises:
        ValueError: If NaN values are detected in the output.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()

    result = np.asarray(X, dtype=np.float32)

    if np.isnan(result).any():
        raise ValueError("NaN values found after dense conversion")

    return result


def build_mlp(n_features: int, n_classes: int) -> tf.keras.Sequential:
    """Build a deterministic MLP for multiclass classification.

    Architecture: Input(n_features) → Dense(256, relu) → Dropout(0.3) →
    Dense(128, relu) → Dropout(0.3) → Dense(64, relu) →
    Dense(n_classes, softmax).

    Args:
        n_features: Number of input features (dynamic, from artifacts).
        n_classes: Number of output classes (dynamic, from LabelEncoder).

    Returns:
        Uncompiled tf.keras.Sequential model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    return model


def compile_mlp(
    model: tf.keras.Sequential,
    learning_rate: float = 0.001,
) -> tf.keras.Sequential:
    """Compile the MLP with Adam, sparse categorical crossentropy, and accuracy.

    Only accuracy is compiled as a Keras metric. Precision, recall, and F1
    are computed post-hoc via sklearn on test predictions (see REQ-08).

    Args:
        model: Uncompiled tf.keras.Sequential from build_mlp().
        learning_rate: Adam learning rate (default 0.001).

    Returns:
        The compiled model (same instance, for chaining).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model
