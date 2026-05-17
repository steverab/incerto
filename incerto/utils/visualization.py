"""
Visualization utilities for examples.

Common plotting functions to avoid duplication across examples.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    title: str = "Training Curves",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot training and validation curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch (optional)
        train_accs: Training accuracies per epoch (optional)
        val_accs: Validation accuracies per epoch (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to call plt.show() (default: True)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss")
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, "b-", label="Train Acc")
    if val_accs is not None:
        axes[1].plot(epochs, val_accs, "r-", label="Val Acc")
    if train_accs is not None or val_accs is not None:
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)

    if show:
        plt.show()


def plot_uncertainty_distribution(
    uncertainties: torch.Tensor,
    correct_mask: torch.Tensor,
    title: str = "Uncertainty Distribution",
    xlabel: str = "Uncertainty",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot uncertainty distribution for correct vs incorrect predictions.

    Args:
        uncertainties: Uncertainty scores (N,)
        correct_mask: Boolean mask for correct predictions (N,)
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure (optional)
        show: Whether to call plt.show() (default: True)
    """
    uncertainties = uncertainties.detach().cpu().numpy()
    correct_mask = correct_mask.detach().cpu().numpy()

    correct_unc = uncertainties[correct_mask]
    incorrect_unc = uncertainties[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    bins = 30
    axes[0].hist(correct_unc, bins=bins, alpha=0.6, label="Correct", color="green")
    axes[0].hist(incorrect_unc, bins=bins, alpha=0.6, label="Incorrect", color="red")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(
        [correct_unc, incorrect_unc],
        labels=["Correct", "Incorrect"],
        patch_artist=True,
    )
    axes[1].set_ylabel(xlabel)
    axes[1].set_title("Box Plot")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)

    if show:
        plt.show()


def plot_2d_classification(
    X: np.ndarray,
    y: np.ndarray,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    title: str = "Classification",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot 2D classification data and decision boundary.

    Args:
        X: Input features (N, 2)
        y: Labels (N,)
        model: Trained model (optional)
        device: Device for model inference
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to call plt.show() (default: True)
    """
    plt.figure(figsize=(10, 8))

    # Plot decision boundary if model provided
    if model is not None:
        model.eval()
        h = 0.02  # Step size in mesh

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
        with torch.no_grad():
            Z = model(grid).cpu().numpy()
            if Z.shape[1] > 1:  # Multi-class
                Z = Z.argmax(axis=1)
            else:  # Binary
                Z = (Z > 0).astype(int).flatten()

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=50)
    plt.colorbar(scatter)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)

    if show:
        plt.show()


__all__ = [
    "plot_training_curves",
    "plot_uncertainty_distribution",
    "plot_2d_classification",
]
