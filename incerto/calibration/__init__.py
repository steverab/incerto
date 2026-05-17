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
    BetaCalibrator,
    DirichletCalibrator,
    HistogramBinningCalibrator,
    IdentityCalibrator,
    IsotonicRegressionCalibrator,
    MatrixScaling,
    PlattScalingCalibrator,
    TemperatureScaling,
    VectorScaling,
)

# Metrics
from .metrics import (
    adaptive_ece_score,
    brier_score,
    classwise_ece,
    ece_score,
    mce_score,
    nll,
    smooth_ece,
)

# Training-time calibration methods
from .training import (
    ConfidencePenalty,
    FocalLoss,
    LabelSmoothingLoss,
    TemperatureAwareTraining,
    evidential_loss,
    get_uncertainty_from_evidence,
)

# Utilities
from .utils import (
    extract_confidences_and_predictions,
    get_bin_stats,
    logits_to_probs,
)

# Visualization
from .visual import (
    plot_calibration_curve,
    plot_confidence_histogram,
    plot_reliability_diagram,
    plot_smooth_reliability_diagram,
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
    "smooth_ece",
    # Visual
    "plot_reliability_diagram",
    "plot_confidence_histogram",
    "plot_calibration_curve",
    "plot_smooth_reliability_diagram",
    # Utils
    "get_bin_stats",
    "extract_confidences_and_predictions",
    "logits_to_probs",
]
