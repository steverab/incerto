"""
Utilities for examples and common use cases.

This module provides:
- Common model architectures
- Training utilities
- Visualization helpers
"""

from .models import ConvNet, ResNet18, MLP
from .training import train_epoch, evaluate, seed_everything
from .visualization import plot_training_curves, plot_uncertainty_distribution

__all__ = [
    # Models
    "ConvNet",
    "ResNet18",
    "MLP",
    # Training
    "train_epoch",
    "evaluate",
    "seed_everything",
    # Visualization
    "plot_training_curves",
    "plot_uncertainty_distribution",
]
