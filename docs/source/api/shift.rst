Distribution Shift Detection
============================

The shift module provides methods for detecting and quantifying distribution shifts
between training and deployment data.

.. currentmodule:: incerto.shift

Statistical Tests
-----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   mmd_test
   energy_distance_test
   ks_test
   classifier_two_sample_test

Label Shift Detection
---------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   label_shift_detection
   bbse_label_shift

Importance Weighting
--------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   importance_weight_estimation

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   wasserstein_distance
   sliced_wasserstein_distance

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_shift_detection
