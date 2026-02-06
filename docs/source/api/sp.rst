Selective Prediction
====================

The selective prediction module enables models to abstain from predictions when uncertain,
providing risk-coverage tradeoffs.

.. currentmodule:: incerto.sp

Base Class
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   BaseSelectivePredictor

Methods
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   SoftmaxThreshold
   SelfAdaptiveTraining
   DeepGambler
   SelectiveNet
   make

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   coverage
   risk
   aurc
   accuracy_coverage_curve

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_risk_coverage
   plot_accuracy_coverage
