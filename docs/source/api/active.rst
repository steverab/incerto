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

   BaseAcquisition
   RandomAcquisition
   EntropyAcquisition
   LeastConfidenceAcquisition
   MarginAcquisition
   BALDAcquisition
   VarianceRatioAcquisition
   MeanSTDAcquisition
   BatchBALDAcquisition

Query Strategies
----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   UncertaintySampling
   DiversitySampling
   CoreSetSelection
   BadgeSampling
   QueryByCommittee

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   split_labeled_unlabeled
   compute_diversity_penalty
   greedy_k_center
   subsample_for_efficiency
   active_learning_loop
