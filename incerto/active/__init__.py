"""
Active Learning for efficient data labeling.

This module provides acquisition functions and query strategies to select
the most informative samples for labeling, reducing annotation costs.

Submodules:
    acquisition: Acquisition functions for scoring unlabeled samples
    strategies: Query strategies for batch selection
    utils: Utility functions for active learning
"""

# Acquisition functions
from .acquisition import (
    BALDAcquisition,
    BaseAcquisition,
    BatchBALDAcquisition,
    EntropyAcquisition,
    LeastConfidenceAcquisition,
    MarginAcquisition,
    MeanSTDAcquisition,
    RandomAcquisition,
    VarianceRatioAcquisition,
)

# Query strategies
from .strategies import (
    BadgeSampling,
    CoreSetSelection,
    DiversitySampling,
    QueryByCommittee,
    UncertaintySampling,
)

# Utilities
from .utils import (
    active_learning_loop,
    compute_diversity_penalty,
    greedy_k_center,
    split_labeled_unlabeled,
    subsample_for_efficiency,
)

__all__ = [
    # Acquisition functions
    "BaseAcquisition",
    "RandomAcquisition",
    "EntropyAcquisition",
    "LeastConfidenceAcquisition",
    "MarginAcquisition",
    "BALDAcquisition",
    "VarianceRatioAcquisition",
    "MeanSTDAcquisition",
    "BatchBALDAcquisition",
    # Query strategies
    "UncertaintySampling",
    "DiversitySampling",
    "CoreSetSelection",
    "BadgeSampling",
    "QueryByCommittee",
    # Utils
    "split_labeled_unlabeled",
    "compute_diversity_penalty",
    "greedy_k_center",
    "subsample_for_efficiency",
    "active_learning_loop",
]
