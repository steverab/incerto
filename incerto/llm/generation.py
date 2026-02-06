"""
Generation-specific uncertainty methods for LLMs.

These methods work with different generation strategies like
beam search, nucleus sampling, etc.
"""

from __future__ import annotations
import torch
from typing import List


class BeamSearchUncertainty:
    """
    Uncertainty estimation from beam search scores.

    Beam search maintains multiple hypotheses with scores.
    The score distribution indicates uncertainty.
    """

    @staticmethod
    def compute_from_scores(
        beam_scores: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict:
        """
        Compute uncertainty from beam search scores.

        Args:
            beam_scores: Scores for each beam (num_beams,)
            temperature: Temperature for softmax (default: 1.0)

        Returns:
            Dictionary with:
                - entropy: Entropy over beam distribution
                - top_beam_prob: Probability of best beam
                - score_variance: Variance of beam scores
        """
        # Convert scores to probabilities
        probs = torch.softmax(beam_scores / temperature, dim=0)

        # Entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum()

        # Top beam probability
        top_beam_prob = probs.max().item()

        # Variance
        score_variance = beam_scores.var().item()

        return {
            "entropy": entropy.item(),
            "top_beam_prob": top_beam_prob,
            "score_variance": score_variance,
            "confidence": top_beam_prob,  # Alias
        }

    @staticmethod
    def diversity_among_beams(
        beam_sequences: List[List[int]],
    ) -> float:
        """
        Measure diversity among beam search outputs.

        Args:
            beam_sequences: List of token ID sequences from beams

        Returns:
            Diversity score (0-1), higher = more diverse
        """
        if len(beam_sequences) < 2:
            return 0.0

        # Compute pairwise differences
        unique_count = len(set(tuple(seq) for seq in beam_sequences))
        max_unique = len(beam_sequences)

        diversity = unique_count / max_unique
        return diversity


class NucleusSamplingUncertainty:
    """
    Uncertainty for nucleus (top-p) sampling.

    Analyzes the probability mass distribution to determine
    how concentrated or spread out the generation is.
    """

    @staticmethod
    def effective_vocabulary_size(
        logits: torch.Tensor,
        p: float = 0.9,
    ) -> int:
        """
        Number of tokens needed to cover p probability mass.

        Args:
            logits: Token logits (..., vocab_size)
            p: Probability mass threshold (default: 0.9)

        Returns:
            Effective vocabulary size
        """
        import torch.nn.functional as F

        probs = F.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumsum = torch.cumsum(sorted_probs, dim=-1)
        nucleus_size = (cumsum <= p).sum(dim=-1) + 1

        return nucleus_size.item() if nucleus_size.dim() == 0 else nucleus_size

    @staticmethod
    def probability_mass_concentration(
        logits: torch.Tensor,
        top_k: int = 10,
    ) -> float:
        """
        Fraction of probability in top-k tokens.

        Args:
            logits: Token logits (..., vocab_size)
            top_k: Number of top tokens

        Returns:
            Probability mass in top-k (0-1)
        """
        import torch.nn.functional as F

        probs = F.softmax(logits, dim=-1)
        top_k_probs, _ = torch.topk(probs, k=top_k, dim=-1)
        mass = top_k_probs.sum(dim=-1)

        return mass.item() if mass.dim() == 0 else mass


class IDontKnowDetection:
    """
    Detect when the model is expressing uncertainty verbally.

    Common patterns: "I don't know", "I'm not sure", "unclear", etc.
    """

    # Common uncertainty phrases
    UNCERTAINTY_PHRASES = [
        "i don't know",
        "i'm not sure",
        "i am not sure",
        "uncertain",
        "unclear",
        "not certain",
        "cannot say",
        "can't say",
        "hard to say",
        "difficult to say",
        "no information",
        "cannot determine",
        "unable to determine",
        "ambiguous",
        "not enough information",
        "insufficient information",
    ]

    @staticmethod
    def contains_uncertainty_phrase(text: str) -> bool:
        """
        Check if text contains uncertainty phrases.

        Args:
            text: Generated text

        Returns:
            True if uncertainty phrase detected
        """
        text_lower = text.lower()
        return any(
            phrase in text_lower for phrase in IDontKnowDetection.UNCERTAINTY_PHRASES
        )

    @staticmethod
    def extract_confidence_from_hedging(text: str) -> dict:
        """
        Detect hedging language and estimate confidence.

        Args:
            text: Generated text

        Returns:
            Dictionary with hedging indicators
        """
        text_lower = text.lower()

        hedges = {
            "maybe": 0.5,
            "perhaps": 0.5,
            "possibly": 0.5,
            "might": 0.6,
            "could": 0.6,
            "probably": 0.7,
            "likely": 0.7,
            "seems": 0.6,
            "appears": 0.6,
            "suggest": 0.6,
            "indicate": 0.6,
        }

        found_hedges = []
        min_confidence = 1.0

        for hedge, confidence in hedges.items():
            if hedge in text_lower:
                found_hedges.append(hedge)
                min_confidence = min(min_confidence, confidence)

        return {
            "hedges_found": found_hedges,
            "num_hedges": len(found_hedges),
            "estimated_confidence": min_confidence if found_hedges else 1.0,
            "contains_hedging": len(found_hedges) > 0,
        }


class ContrastiveDecoding:
    """
    Uncertainty from contrastive decoding (comparing expert vs amateur models).

    Uses the difference in predictions between a strong and weak model
    to identify regions of high uncertainty.
    """

    @staticmethod
    def compute_contrastive_score(
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute contrastive decoding score.

        Score = expert_prob - alpha * amateur_prob

        Args:
            expert_logits: Logits from expert/strong model
            amateur_logits: Logits from amateur/weak model
            alpha: Weight for amateur contribution

        Returns:
            Contrastive scores
        """
        import torch.nn.functional as F

        expert_probs = F.softmax(expert_logits, dim=-1)
        amateur_probs = F.softmax(amateur_logits, dim=-1)

        contrastive_scores = expert_probs - alpha * amateur_probs

        return contrastive_scores

    @staticmethod
    def disagreement_score(
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Measure disagreement between expert and amateur.

        Args:
            expert_logits: Logits from expert model
            amateur_logits: Logits from amateur model

        Returns:
            Disagreement score (KL divergence)
        """
        import torch.nn.functional as F

        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        amateur_probs = F.softmax(amateur_logits, dim=-1)

        # KL divergence: D_KL(amateur || expert)
        kl_div = F.kl_div(expert_log_probs, amateur_probs, reduction="none")
        disagreement = kl_div.sum(dim=-1)

        return disagreement
