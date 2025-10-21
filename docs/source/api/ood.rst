Out-of-Distribution Detection
==============================

The OOD module provides methods for detecting when a model encounters data that is
significantly different from its training distribution.

.. currentmodule:: incerto.ood

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
   cutmix_data
   OutlierExposureLoss
   EnergyRegularizationLoss

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   auroc
   aupr
   fpr_at_tpr

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_ood_histogram
   plot_roc_curve
