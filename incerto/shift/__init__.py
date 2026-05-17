"""
Distribution shift detection methods, metrics, and visualizations.
"""

# Base classes
from .base import BaseShiftDetector

# Shift detection methods
from .methods import (
    BBSDDetector,
    ClassifierShiftDetector,
    EnergyShiftDetector,
    ImportanceWeightingShift,
    KSShiftDetector,
    LabelShiftDetector,
    MMDShiftDetector,
)

# Metrics
from .metrics import (
    energy_distance,
    population_stability_index,
    sliced_wasserstein_distance,
    total_variation,
    wasserstein_distance,
)

# Visualization
from .visual import (
    plot_confidence_distributions,
    plot_embedding_space,
    plot_feature_histograms,
    plot_ks_statistics,
    plot_shift_severity,
)

__all__ = [
    # Base
    "BaseShiftDetector",
    # Methods
    "MMDShiftDetector",
    "EnergyShiftDetector",
    "KSShiftDetector",
    "ClassifierShiftDetector",
    "BBSDDetector",
    "LabelShiftDetector",
    "ImportanceWeightingShift",
    # Metrics
    "energy_distance",
    "total_variation",
    "population_stability_index",
    "wasserstein_distance",
    "sliced_wasserstein_distance",
    # Visual
    "plot_feature_histograms",
    "plot_embedding_space",
    "plot_confidence_distributions",
    "plot_shift_severity",
    "plot_ks_statistics",
]
