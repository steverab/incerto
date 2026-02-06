Out-of-Distribution Detection
==============================

The OOD module provides methods for detecting when a model encounters data that is
significantly different from its training distribution.

.. currentmodule:: incerto.ood

Base Class
----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   OODDetector

Score-based Methods
-------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   MSP
   MaxLogit
   Energy
   ODIN

Distance-based Methods
----------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   Mahalanobis
   KNN

Training Methods
----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   mixup_data
   mixup_criterion
   OutlierExposureLoss
   EnergyRegularizedLoss
   CutMix

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   auroc
   fpr_at_tpr
   detection_accuracy

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_roc
   score_hist

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   compute_threshold_at_tpr
   get_ood_predictions
   extract_features
