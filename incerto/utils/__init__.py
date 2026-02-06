"""
Utilities for examples and common use cases.

This module provides:
- Common model architectures
- Training utilities
- Visualization helpers
- Logging configuration
"""

from .models import ConvNet, ResNet18, MLP, BasicBlock
from .training import train_epoch, evaluate, seed_everything, EarlyStopping
from .visualization import (
    plot_training_curves,
    plot_uncertainty_distribution,
    plot_2d_classification,
)
from .logging import get_logger, setup_logging, disable_logging

__all__ = [
    # Models
    "ConvNet",
    "ResNet18",
    "MLP",
    "BasicBlock",
    # Training
    "train_epoch",
    "evaluate",
    "seed_everything",
    "EarlyStopping",
    # Visualization
    "plot_training_curves",
    "plot_uncertainty_distribution",
    "plot_2d_classification",
    # Logging
    "get_logger",
    "setup_logging",
    "disable_logging",
]
