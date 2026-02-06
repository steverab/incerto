Conformal Prediction
====================

The conformal prediction module provides distribution-free uncertainty quantification
with finite-sample coverage guarantees.

.. currentmodule:: incerto.conformal

Classification Methods
----------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   inductive_conformal
   mondrian_conformal
   aps
   raps

Regression Methods
------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   jackknife_plus
   cv_plus
   conformalized_quantile_regression

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   empirical_coverage
   average_set_size
   conditional_coverage

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_coverage_vs_alpha
   plot_set_size_hist

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ConformalPredictor
   compute_quantile
   prediction_set_from_scores
   split_data
