Distribution Shift Detection
============================

The shift module provides methods for detecting and quantifying distribution shifts
between training and deployment data.

.. currentmodule:: incerto.shift

Base Class
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   BaseShiftDetector

Shift Detectors
---------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   MMDShiftDetector
   EnergyShiftDetector
   KSShiftDetector
   ClassifierShiftDetector
   BBSDDetector
   LabelShiftDetector
   ImportanceWeightingShift

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   energy_distance
   total_variation
   population_stability_index
   wasserstein_distance
   sliced_wasserstein_distance

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_feature_histograms
   plot_embedding_space
   plot_confidence_distributions
   plot_shift_severity
   plot_ks_statistics
