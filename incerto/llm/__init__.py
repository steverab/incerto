"""
LLM Uncertainty Quantification Module.

This module provides comprehensive uncertainty quantification methods
specifically designed for Large Language Models (LLMs).

Submodules:
    token: Token-level uncertainty measures
    sequence: Sequence-level uncertainty aggregation
    sampling: Sampling-based uncertainty (multiple generations)
    verbalized: Prompting-based uncertainty elicitation
    calibration: Calibration methods for LLM outputs
    metrics: Evaluation metrics
    visual: Visualization utilities
"""

# Token-level uncertainty
from .token import (
    TokenEntropy,
    TokenConfidence,
    TokenPerplexity,
    SurprisalScore,
    TopKConfidence,
)

# Sequence-level uncertainty
from .sequence import (
    SequenceProbability,
    AverageLogProb,
    NormalizedSequenceProb,
    SequenceEntropy,
    SequencePerplexity,
    VarianceOfTokenProbs,
)

# Sampling-based uncertainty
from .sampling import (
    SelfConsistency,
    LexicalSimilarity,
    VarianceRatio,
    PredictiveEntropy,
    MutualInformation,
    SemanticEntropy,
    EnsembleDisagreement,
)

# Verbalized uncertainty
from .verbalized import (
    VerbalizedConfidence,
    PTrue,
    SelfEvaluation,
    BidirectionalConsistency,
)

# Calibration
from .calibration import (
    TokenTemperatureScaling,
    SequenceLengthCalibration,
    VerbosityBiasCorrection,
    HistogramBinning,
)

# Metrics
from .metrics import (
    selective_accuracy,
    calibration_error,
    brier_score,
    aur_c,
    uncertainty_auc,
    token_level_accuracy,
    sequence_level_accuracy,
    f1_score_tokens,
)

# Visualization
from .visual import (
    plot_token_uncertainty,
    plot_confidence_vs_correctness,
    plot_generation_diversity,
    plot_semantic_clusters,
    plot_risk_coverage_llm,
    plot_uncertainty_distribution,
    plot_length_vs_confidence,
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
