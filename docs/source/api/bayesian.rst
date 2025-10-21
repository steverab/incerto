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

   MCDropout
   DeepEnsemble
   SWAG
   LaplaceApproximation
   VariationalInference

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   predictive_entropy
   mutual_information
   expected_pairwise_kl
   disagreement

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   decompose_uncertainty
