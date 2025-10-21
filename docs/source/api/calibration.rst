Calibration
===========

The calibration module provides methods for calibrating neural network predictions to ensure
that confidence scores accurately reflect the true probability of correctness.

.. currentmodule:: incerto.calibration

Post-hoc Calibration Methods
-----------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   TemperatureScaling
   VectorScaling
   MatrixScaling
   PlattScaling
   IsotonicRegression
   HistogramBinning
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
   EvidentialLoss
   TemperatureAwareTraining

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ece_score
   mce_score
   classwise_ece_score
   adaptive_ece_score
   brier_score
   nll_score

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_reliability_diagram
   plot_confidence_histogram
   plot_calibration_curve

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   PredictionDistribution
