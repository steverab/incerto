"""
Selective prediction methods, metrics, and visualizations.
"""

# Base classes
from .base import BaseSelectivePredictor

# Selective prediction methods
from .methods import (
    DeepGambler,
    SelectiveNet,
    SelfAdaptiveTraining,
    SoftmaxThreshold,
    make,
)

# Metrics
from .metrics import (
    accuracy_coverage_curve,
    aurc,
    coverage,
    risk,
)

# Visualization
from .visual import (
    plot_accuracy_coverage,
    plot_risk_coverage,
)

__all__ = [
    # Base
    "BaseSelectivePredictor",
    # Methods
    "SoftmaxThreshold",
    "DeepGambler",
    "SelectiveNet",
    "SelfAdaptiveTraining",
    "make",
    # Metrics
    "coverage",
    "risk",
    "aurc",
    "accuracy_coverage_curve",
    # Visual
    "plot_risk_coverage",
    "plot_accuracy_coverage",
]
