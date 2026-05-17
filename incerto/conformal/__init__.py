"""
Conformal prediction methods, metrics, and visualizations.
"""

# Conformal methods
from .methods import (
    ConformalPredictor,
    aps,
    conformalized_quantile_regression,
    cv_plus,
    inductive_conformal,
    jackknife_plus,
    mondrian_conformal,
    raps,
)

# Metrics
from .metrics import (
    average_set_size,
    conditional_coverage,
    empirical_coverage,
)

# Utilities
from .utils import (
    compute_quantile,
    prediction_set_from_scores,
    split_data,
)

# Visualization
from .visual import (
    plot_coverage_vs_alpha,
    plot_set_size_hist,
)

__all__ = [
    # Methods
    "ConformalPredictor",
    "inductive_conformal",
    "mondrian_conformal",
    "aps",
    "raps",
    "jackknife_plus",
    "cv_plus",
    "conformalized_quantile_regression",
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
