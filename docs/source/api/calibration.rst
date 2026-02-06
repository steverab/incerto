Calibration
===========

The calibration module provides methods for calibrating neural network predictions to ensure
that confidence scores accurately reflect the true probability of correctness.

.. currentmodule:: incerto.calibration

Base Class
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   BaseCalibrator

Post-hoc Calibration Methods
-----------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   TemperatureScaling
   VectorScaling
   MatrixScaling
   PlattScalingCalibrator
   IsotonicRegressionCalibrator
   HistogramBinningCalibrator
   DirichletCalibrator
   BetaCalibrator
   IdentityCalibrator

Training-time Calibration
-------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   LabelSmoothingLoss
   FocalLoss
   ConfidencePenalty
   evidential_loss
   get_uncertainty_from_evidence
   TemperatureAwareTraining

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ece_score
   mce_score
   classwise_ece
   adaptive_ece_score
   smooth_ece
   brier_score
   nll

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_reliability_diagram
   plot_smooth_reliability_diagram
   plot_confidence_histogram
   plot_calibration_curve

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   get_bin_stats
   extract_confidences_and_predictions
   logits_to_probs
