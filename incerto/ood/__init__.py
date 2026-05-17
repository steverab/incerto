"""
Out-of-distribution detection methods, metrics, and visualizations.

This module provides:
- Post-hoc OOD detection methods (methods.py): Applied after training
- Training-time OOD methods (training.py): Integrated into training
- OOD detection metrics and visualizations
"""

# Base classes
from .base import OODDetector

# Post-hoc OOD detection methods
from .methods import (
    KNN,
    MSP,
    ODIN,
    Energy,
    Mahalanobis,
    MaxLogit,
)

# Metrics
from .metrics import (
    auroc,
    detection_accuracy,
    fpr_at_tpr,
)

# Training-time OOD methods
from .training import (
    CutMix,
    EnergyRegularizedLoss,
    OutlierExposureLoss,
    mixup_criterion,
    mixup_data,
)

# Utilities
from .utils import (
    compute_threshold_at_tpr,
    extract_features,
    get_ood_predictions,
)

# Visualization
from .visual import (
    plot_roc,
    score_hist,
)

__all__ = [
    # Base
    "OODDetector",
    # Post-hoc methods
    "MSP",
    "Energy",
    "ODIN",
    "Mahalanobis",
    "MaxLogit",
    "KNN",
    # Training-time methods
    "mixup_data",
    "mixup_criterion",
    "OutlierExposureLoss",
    "EnergyRegularizedLoss",
    "CutMix",
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
