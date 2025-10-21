LLM Uncertainty
===============

The LLM module provides uncertainty quantification methods specifically designed for
large language models.

.. currentmodule:: incerto.llm

Token-level Uncertainty
-----------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   TokenEntropy
   TokenConfidence
   Perplexity
   SurprisalScore
   TopKConfidence

Sequence-level Uncertainty
--------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   SequenceProbability
   AverageLogProbability
   SequenceEntropy

Sampling-based Uncertainty
--------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   SelfConsistency
   SemanticEntropy
   PredictiveEntropy
   MutualInformation

Generation Methods
------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   BeamSearchUncertainty
   NucleusSampling
   IDontKnowDetection
   ContrastiveDecoding

Calibration
-----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   AnswerLevelCalibration

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_token_confidence
   plot_sequence_uncertainty
