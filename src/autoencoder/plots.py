"""Headless plotting for the experimental autoencoder module.

All plots render to PNG without a display server via the Agg backend.
Functions auto-create output directories if missing.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def plot_training_loss(history, out_path: str) -> None:
    """Plot training and validation loss curves over epochs.

    Args:
        history: A tf.keras.callbacks.History object (or any object with
            a .history dict containing 'loss' and optionally 'val_loss').
        out_path: File path for the output PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss) + 1), loss, label="Training Loss", linewidth=2)

    if val_loss:
        plt.plot(
            range(1, len(val_loss) + 1),
            val_loss,
            label="Validation Loss",
            linewidth=2,
            linestyle="--",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_reconstruction_error_distribution(
    errors: np.ndarray, threshold: float, out_path: str
) -> None:
    """Plot histogram of reconstruction errors with a threshold line.

    Args:
        errors: 1D array of reconstruction errors (n_samples,).
        threshold: Anomaly threshold value to draw as a vertical line.
        out_path: File path for the output PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.6f}",
    )

    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str,
    out_path: str,
) -> None:
    """Plot a confusion matrix as a heatmap with class names.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: List of class name strings matching label order.
        title: Title for the plot.
        out_path: File path for the output PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(label="Count")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
