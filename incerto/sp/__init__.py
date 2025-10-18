"""
Selective prediction methods, metrics, and visualizations.
"""

# Base classes
from .base import BaseSelectivePredictor

# Selective prediction methods
from .methods import (
    SoftmaxThreshold,
    DeepGambler,
    SelectiveNet,
    SelfAdaptiveTraining,
    make,
)

# Metrics
from .metrics import (
    coverage,
    risk,
    aurc,
    accuracy_coverage_curve,
)

# Visualization
from .visual import (
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
]
