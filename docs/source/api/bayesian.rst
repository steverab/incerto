Bayesian Deep Learning
======================

The Bayesian module provides methods for approximate Bayesian inference in neural networks,
enabling uncertainty decomposition into epistemic and aleatoric components.

.. currentmodule:: incerto.bayesian

Methods
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   BaseBayesianMethod
   MCDropout
   DeepEnsemble
   SWAG
   LaplaceApproximation
   VariationalBayesNN
   GaussianLinear

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ensemble_diversity
   uncertainty_quality
   disagreement
   negative_log_likelihood
   brier_score
   predictive_log_likelihood
   sharpness

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   predictive_entropy
   mutual_information
   expected_calibration_error
   decompose_uncertainty
   compute_disagreement
   sample_from_posterior
   ensemble_predictions_to_distribution
