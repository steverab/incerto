"""
Out-of-distribution detection methods, metrics, and visualizations.
"""

# Base classes
from .base import OODDetector

# OOD detection methods
from .methods import (
    MSP,
    Energy,
    ODIN,
    Mahalanobis,
    MaxLogit,
    KNN,
)

# Metrics
from .metrics import (
    auroc,
    fpr_at_tpr,
    detection_accuracy,
)

# Visualization
from .visual import (
    plot_roc,
    score_hist,
)

# Utilities
from .utils import (
    compute_threshold_at_tpr,
    get_ood_predictions,
    extract_features,
)

__all__ = [
    # Base
    "OODDetector",
    # Methods
    "MSP",
    "Energy",
    "ODIN",
    "Mahalanobis",
    "MaxLogit",
    "KNN",
    # Metrics
    "auroc",
    "fpr_at_tpr",
    "detection_accuracy",
    # Visual
    "plot_roc",
    "score_hist",
    # Utils
    "compute_threshold_at_tpr",
    "get_ood_predictions",
    "extract_features",
]
