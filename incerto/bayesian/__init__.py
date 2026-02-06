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
    BaseBayesianMethod,
    MCDropout,
    DeepEnsemble,
    SWAG,
    LaplaceApproximation,
    VariationalBayesNN,
    GaussianLinear,
)

# Utilities
from .utils import (
    predictive_entropy,
    mutual_information,
    expected_calibration_error,
    decompose_uncertainty,
    compute_disagreement,
    sample_from_posterior,
    ensemble_predictions_to_distribution,
)

# Metrics
from .metrics import (
    ensemble_diversity,
    uncertainty_quality,
    disagreement,
    negative_log_likelihood,
    brier_score,
    predictive_log_likelihood,
    sharpness,
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
