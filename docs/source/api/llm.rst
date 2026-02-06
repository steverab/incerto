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
   TokenPerplexity
   SurprisalScore
   TopKConfidence

Sequence-level Uncertainty
--------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   SequenceProbability
   AverageLogProb
   NormalizedSequenceProb
   SequenceEntropy
   SequencePerplexity
   VarianceOfTokenProbs

Sampling-based Uncertainty
--------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   SelfConsistency
   LexicalSimilarity
   VarianceRatio
   PredictiveEntropy
   MutualInformation
   SemanticEntropy
   EnsembleDisagreement

Generation Methods
------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   BeamSearchUncertainty
   NucleusSamplingUncertainty
   IDontKnowDetection
   ContrastiveDecoding

Verbalized Uncertainty
----------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   VerbalizedConfidence
   PTrue
   SelfEvaluation
   BidirectionalConsistency

Calibration
-----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   TokenTemperatureScaling
   SequenceLengthCalibration
   VerbosityBiasCorrection
   HistogramBinning

Metrics
-------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   selective_accuracy
   calibration_error
   brier_score
   aur_c
   uncertainty_auc
   token_level_accuracy
   sequence_level_accuracy
   f1_score_tokens

Visualization
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   plot_token_uncertainty
   plot_confidence_vs_correctness
   plot_generation_diversity
   plot_semantic_clusters
   plot_risk_coverage_llm
   plot_uncertainty_distribution
   plot_length_vs_confidence
