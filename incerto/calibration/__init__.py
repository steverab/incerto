"""
Calibration methods, metrics, and visualizations.

This module provides:
- Post-hoc calibration methods (methods.py): Applied after training
- Training-time calibration methods (training.py): Integrated into training
- Calibration metrics and visualizations
"""

# Base classes
from .base import BaseCalibrator

# Post-hoc calibration methods
from .methods import (
    IdentityCalibrator,
    TemperatureScaling,
    IsotonicRegressionCalibrator,
    HistogramBinningCalibrator,
    PlattScalingCalibrator,
    VectorScaling,
    MatrixScaling,
    DirichletCalibrator,
    BetaCalibrator,
)

# Training-time calibration methods
from .training import (
    LabelSmoothingLoss,
    FocalLoss,
    ConfidencePenalty,
    evidential_loss,
    get_uncertainty_from_evidence,
    TemperatureAwareTraining,
)

# Metrics
from .metrics import (
    nll,
    brier_score,
    ece_score,
    mce_score,
    classwise_ece,
    adaptive_ece_score,
)

# Visualization
from .visual import (
    plot_reliability_diagram,
    plot_confidence_histogram,
    plot_calibration_curve,
)

# Utilities
from .utils import (
    get_bin_stats,
    extract_confidences_and_predictions,
    logits_to_probs,
)

__all__ = [
    # Base
    "BaseCalibrator",
    # Post-hoc methods
    "IdentityCalibrator",
    "TemperatureScaling",
    "IsotonicRegressionCalibrator",
    "HistogramBinningCalibrator",
    "PlattScalingCalibrator",
    "VectorScaling",
    "MatrixScaling",
    "DirichletCalibrator",
    "BetaCalibrator",
    # Training-time methods
    "LabelSmoothingLoss",
    "FocalLoss",
    "ConfidencePenalty",
    "evidential_loss",
    "get_uncertainty_from_evidence",
    "TemperatureAwareTraining",
    # Metrics
    "nll",
    "brier_score",
    "ece_score",
    "mce_score",
    "classwise_ece",
    "adaptive_ece_score",
    # Visual
    "plot_reliability_diagram",
    "plot_confidence_histogram",
    "plot_calibration_curve",
    # Utils
    "get_bin_stats",
    "extract_confidences_and_predictions",
    "logits_to_probs",
]
