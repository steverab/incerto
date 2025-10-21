Selective Prediction
====================

The selective prediction module enables models to abstain from predictions when uncertain,
providing risk-coverage tradeoffs.

.. currentmodule:: incerto.sp

Methods
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   SoftmaxThreshold
   SelfAdaptiveTraining
   DeepGambler
   SelectiveNet

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   selective_risk
   coverage_at_risk
   aurc
   eaurc

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_risk_coverage_curve
