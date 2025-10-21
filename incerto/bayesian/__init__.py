"""
Bayesian Deep Learning for Uncertainty Quantification.

This module provides Bayesian approaches to deep learning that naturally
quantify both epistemic (model) and aleatoric (data) uncertainty.

Submodules:
    methods: Core Bayesian DL methods (MC Dropout, Deep Ensembles, etc.)
    metrics: Metrics for evaluating Bayesian predictions
    utils: Utility functions for Bayesian inference
"""

# Core methods
from .methods import (
    MCDropout,
    DeepEnsemble,
    SWAG,
    LaplaceApproximation,
    VariationalBayesNN,
)

# Utilities
from .utils import (
    predictive_entropy,
    mutual_information,
    expected_calibration_error,
    decompose_uncertainty,
)

# Metrics
from .metrics import (
    ensemble_diversity,
    uncertainty_quality,
    disagreement,
)

__all__ = [
    # Methods
    "MCDropout",
    "DeepEnsemble",
    "SWAG",
    "LaplaceApproximation",
    "VariationalBayesNN",
    # Utils
    "predictive_entropy",
    "mutual_information",
    "expected_calibration_error",
    "decompose_uncertainty",
    # Metrics
    "ensemble_diversity",
    "uncertainty_quality",
    "disagreement",
]
