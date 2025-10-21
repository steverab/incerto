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
    RandomAcquisition,
    EntropyAcquisition,
    LeastConfidenceAcquisition,
    MarginAcquisition,
    BALDAcquisition,
    VarianceRatioAcquisition,
    MeanSTDAcquisition,
    BatchBALDAcquisition,
)

# Query strategies
from .strategies import (
    UncertaintySampling,
    DiversitySampling,
    CoreSetSelection,
    BadgeSampling,
    QueryByCommittee,
)

# Utilities
from .utils import (
    split_labeled_unlabeled,
    compute_diversity_penalty,
    greedy_k_center,
)

__all__ = [
    # Acquisition functions
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
]
