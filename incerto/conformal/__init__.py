"""
Conformal prediction methods, metrics, and visualizations.
"""

# Conformal methods
from .methods import (
    inductive_conformal,
    mondrian_conformal,
    aps,
    raps,
    jackknife_plus,
    cv_plus,
)

# Metrics
from .metrics import (
    empirical_coverage,
    average_set_size,
    conditional_coverage,
)

# Visualization
from .visual import (
    plot_coverage_vs_alpha,
    plot_set_size_hist,
)

# Utilities
from .utils import (
    compute_quantile,
    prediction_set_from_scores,
    split_data,
)

__all__ = [
    # Methods
    "inductive_conformal",
    "mondrian_conformal",
    "aps",
    "raps",
    "jackknife_plus",
    "cv_plus",
    # Metrics
    "empirical_coverage",
    "average_set_size",
    "conditional_coverage",
    # Visual
    "plot_coverage_vs_alpha",
    "plot_set_size_hist",
    # Utils
    "compute_quantile",
    "prediction_set_from_scores",
    "split_data",
]
