"""
Distribution shift detection methods, metrics, and visualizations.
"""

# Base classes
from .base import BaseShiftDetector

# Shift detection methods
from .methods import (
    MMDShiftDetector,
    EnergyShiftDetector,
    KSShiftDetector,
    ClassifierShiftDetector,
)

# Metrics
from .metrics import (
    energy_distance,
    total_variation,
    population_stability_index,
)

# Visualization
from .visual import (
    plot_feature_histograms,
    plot_embedding_space,
)

__all__ = [
    # Base
    "BaseShiftDetector",
    # Methods
    "MMDShiftDetector",
    "EnergyShiftDetector",
    "KSShiftDetector",
    "ClassifierShiftDetector",
    # Metrics
    "energy_distance",
    "total_variation",
    "population_stability_index",
    # Visual
    "plot_feature_histograms",
    "plot_embedding_space",
]
