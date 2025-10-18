"""
Calibration methods, metrics, and visualizations.
"""

# Base classes
from .base import BaseCalibrator

# Calibration methods
from .methods import (
    IdentityCalibrator,
    TemperatureScaling,
    IsotonicRegressionCalibrator,
    HistogramBinningCalibrator,
    PlattScalingCalibrator,
    VectorScaling,
    MatrixScaling,
)

# Metrics
from .metrics import (
    nll,
    brier_score,
    ece_score,
    mce_score,
    classwise_ece,
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
    # Methods
    "IdentityCalibrator",
    "TemperatureScaling",
    "IsotonicRegressionCalibrator",
    "HistogramBinningCalibrator",
    "PlattScalingCalibrator",
    "VectorScaling",
    "MatrixScaling",
    # Metrics
    "nll",
    "brier_score",
    "ece_score",
    "mce_score",
    "classwise_ece",
    # Visual
    "plot_reliability_diagram",
    "plot_confidence_histogram",
    "plot_calibration_curve",
    # Utils
    "get_bin_stats",
    "extract_confidences_and_predictions",
    "logits_to_probs",
]
