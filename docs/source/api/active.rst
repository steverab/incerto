Active Learning
===============

The active learning module provides acquisition functions and query strategies for
efficiently selecting the most informative samples to label.

.. currentmodule:: incerto.active

Acquisition Functions
---------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   entropy_acquisition
   bald_acquisition
   margin_acquisition
   variance_ratio_acquisition
   mean_std_acquisition
   least_confidence_acquisition
   max_entropy_acquisition
   random_acquisition

Query Strategies
----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   UncertaintySampling
   DiversitySampling
   CoreSet
   BADGE
   BatchBALD

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ActiveLearningDataset
