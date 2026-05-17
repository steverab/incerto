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
    SWAG,
    BaseBayesianMethod,
    DeepEnsemble,
    GaussianLinear,
    LaplaceApproximation,
    MCDropout,
    VariationalBayesNN,
)

# Metrics
from .metrics import (
    brier_score,
    disagreement,
    ensemble_diversity,
    negative_log_likelihood,
    predictive_log_likelihood,
    sharpness,
    uncertainty_quality,
)

# Utilities
from .utils import (
    compute_disagreement,
    decompose_uncertainty,
    ensemble_predictions_to_distribution,
    expected_calibration_error,
    mutual_information,
    predictive_entropy,
    sample_from_posterior,
)

__all__ = [
    # Methods
    "BaseBayesianMethod",
    "MCDropout",
    "DeepEnsemble",
    "SWAG",
    "LaplaceApproximation",
    "VariationalBayesNN",
    "GaussianLinear",
    # Utils
    "predictive_entropy",
    "mutual_information",
    "expected_calibration_error",
    "decompose_uncertainty",
    "compute_disagreement",
    "sample_from_posterior",
    "ensemble_predictions_to_distribution",
    # Metrics
    "ensemble_diversity",
    "uncertainty_quality",
    "disagreement",
    "negative_log_likelihood",
    "brier_score",
    "predictive_log_likelihood",
    "sharpness",
]
