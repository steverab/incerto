"""
LLM Uncertainty Quantification Module.

This module provides comprehensive uncertainty quantification methods
specifically designed for Large Language Models (LLMs).

Submodules:
    token: Token-level uncertainty measures
    sequence: Sequence-level uncertainty aggregation
    sampling: Sampling-based uncertainty (multiple generations)
    generation: Generation-specific methods (beam search, nucleus sampling)
    verbalized: Prompting-based uncertainty elicitation
    calibration: Calibration methods for LLM outputs
    metrics: Evaluation metrics
    visual: Visualization utilities
"""

# Token-level uncertainty
# Calibration
from .calibration import (
    HistogramBinning,
    SequenceLengthCalibration,
    TokenTemperatureScaling,
    VerbosityBiasCorrection,
)

# Generation-specific uncertainty
from .generation import (
    BeamSearchUncertainty,
    ContrastiveDecoding,
    IDontKnowDetection,
    NucleusSamplingUncertainty,
)

# Metrics
from .metrics import (
    aur_c,
    brier_score,
    calibration_error,
    f1_score_tokens,
    selective_accuracy,
    sequence_level_accuracy,
    token_level_accuracy,
    uncertainty_auc,
)

# Sampling-based uncertainty
from .sampling import (
    EnsembleDisagreement,
    LexicalSimilarity,
    MutualInformation,
    PredictiveEntropy,
    SelfConsistency,
    SemanticEntropy,
    VarianceRatio,
)

# Sequence-level uncertainty
from .sequence import (
    AverageLogProb,
    NormalizedSequenceProb,
    SequenceEntropy,
    SequencePerplexity,
    SequenceProbability,
    VarianceOfTokenProbs,
)
from .token import (
    SurprisalScore,
    TokenConfidence,
    TokenEntropy,
    TokenPerplexity,
    TopKConfidence,
)

# Verbalized uncertainty
from .verbalized import (
    BidirectionalConsistency,
    PTrue,
    SelfEvaluation,
    VerbalizedConfidence,
)

# Visualization
from .visual import (
    plot_confidence_vs_correctness,
    plot_generation_diversity,
    plot_length_vs_confidence,
    plot_risk_coverage_llm,
    plot_semantic_clusters,
    plot_token_uncertainty,
    plot_uncertainty_distribution,
)

__all__ = [
    # Token-level
    "TokenEntropy",
    "TokenConfidence",
    "TokenPerplexity",
    "SurprisalScore",
    "TopKConfidence",
    # Sequence-level
    "SequenceProbability",
    "AverageLogProb",
    "NormalizedSequenceProb",
    "SequenceEntropy",
    "SequencePerplexity",
    "VarianceOfTokenProbs",
    # Sampling-based
    "SelfConsistency",
    "LexicalSimilarity",
    "VarianceRatio",
    "PredictiveEntropy",
    "MutualInformation",
    "SemanticEntropy",
    "EnsembleDisagreement",
    # Generation-specific
    "BeamSearchUncertainty",
    "NucleusSamplingUncertainty",
    "IDontKnowDetection",
    "ContrastiveDecoding",
    # Verbalized
    "VerbalizedConfidence",
    "PTrue",
    "SelfEvaluation",
    "BidirectionalConsistency",
    # Calibration
    "TokenTemperatureScaling",
    "SequenceLengthCalibration",
    "VerbosityBiasCorrection",
    "HistogramBinning",
    # Metrics
    "selective_accuracy",
    "calibration_error",
    "brier_score",
    "aur_c",
    "uncertainty_auc",
    "token_level_accuracy",
    "sequence_level_accuracy",
    "f1_score_tokens",
    # Visualization
    "plot_token_uncertainty",
    "plot_confidence_vs_correctness",
    "plot_generation_diversity",
    "plot_semantic_clusters",
    "plot_risk_coverage_llm",
    "plot_uncertainty_distribution",
    "plot_length_vs_confidence",
]
