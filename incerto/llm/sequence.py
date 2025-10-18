"""
Sequence-level uncertainty quantification for LLMs.

These methods aggregate token-level uncertainties to produce
a single uncertainty score for an entire generated sequence.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from .token import TokenEntropy, SurprisalScore


class SequenceProbability:
    """
    Joint probability of the entire sequence.

    Computed as the product of individual token probabilities.
    Lower values indicate higher uncertainty.
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute sequence probability.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            token_ids: Generated token IDs of shape (batch, seq_len)
            dim: Dimension to compute softmax over (default: -1)

        Returns:
            Sequence probabilities of shape (batch,)
        """
        log_probs = F.log_softmax(logits, dim=dim)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probs = product of probs
        seq_log_prob = token_log_probs.sum(dim=1)
        seq_prob = torch.exp(seq_log_prob)

        return seq_prob


class AverageLogProb:
    """
    Mean log-probability across the sequence.

    Average negative log-likelihood per token. Higher (less negative)
    values indicate lower uncertainty.
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute average log probability.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            token_ids: Generated token IDs of shape (batch, seq_len)
            mask: Optional mask for padding of shape (batch, seq_len)
            dim: Dimension to compute softmax over (default: -1)

        Returns:
            Average log probabilities of shape (batch,)
        """
        log_probs = F.log_softmax(logits, dim=dim)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)

        if mask is not None:
            # Apply mask and normalize by actual length
            token_log_probs = token_log_probs * mask
            seq_lengths = mask.sum(dim=1)
            avg_log_prob = token_log_probs.sum(dim=1) / seq_lengths
        else:
            avg_log_prob = token_log_probs.mean(dim=1)

        return avg_log_prob


class NormalizedSequenceProb:
    """
    Length-normalized sequence probability.

    Accounts for the fact that longer sequences naturally have
    lower probabilities. Common in beam search and generation.
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        length_penalty: float = 1.0,
        mask: torch.Tensor | None = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute length-normalized sequence probability.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            token_ids: Generated token IDs of shape (batch, seq_len)
            length_penalty: Length normalization factor (higher = more penalty)
            mask: Optional mask for padding of shape (batch, seq_len)
            dim: Dimension to compute softmax over (default: -1)

        Returns:
            Normalized probabilities of shape (batch,)
        """
        avg_log_prob = AverageLogProb.compute(logits, token_ids, mask, dim)

        if mask is not None:
            seq_lengths = mask.sum(dim=1)
        else:
            seq_lengths = torch.tensor(
                token_ids.size(1), dtype=torch.float, device=token_ids.device
            ).expand(token_ids.size(0))

        # Length normalization: score / length^penalty
        normalized_score = avg_log_prob / (seq_lengths**length_penalty)

        return torch.exp(normalized_score)


class SequenceEntropy:
    """
    Aggregated entropy over the sequence.

    Can use mean, sum, or max aggregation of token-level entropies.
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        aggregation: str = "mean",
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute sequence entropy.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            mask: Optional mask for padding of shape (batch, seq_len)
            aggregation: How to aggregate ("mean", "sum", "max")
            dim: Dimension to compute entropy over (default: -1)

        Returns:
            Sequence entropy of shape (batch,)
        """
        # Compute token-level entropy
        token_entropy = TokenEntropy.compute(logits, dim=dim)

        if mask is not None:
            token_entropy = token_entropy * mask

        # Aggregate
        if aggregation == "mean":
            if mask is not None:
                seq_lengths = mask.sum(dim=1)
                result = token_entropy.sum(dim=1) / seq_lengths
            else:
                result = token_entropy.mean(dim=1)
        elif aggregation == "sum":
            result = token_entropy.sum(dim=1)
        elif aggregation == "max":
            if mask is not None:
                # Set masked positions to -inf before max
                token_entropy = token_entropy.masked_fill(~mask.bool(), float("-inf"))
            result, _ = token_entropy.max(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return result


class SequencePerplexity:
    """
    Perplexity of the entire sequence.

    Exponential of average log-probability. Standard metric
    for language model quality.
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute sequence perplexity.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            token_ids: Generated token IDs of shape (batch, seq_len)
            mask: Optional mask for padding of shape (batch, seq_len)
            dim: Dimension to compute softmax over (default: -1)

        Returns:
            Perplexity values of shape (batch,)
        """
        avg_log_prob = AverageLogProb.compute(logits, token_ids, mask, dim)
        # Perplexity = exp(-log_prob) = exp(negative log likelihood)
        perplexity = torch.exp(-avg_log_prob)
        return perplexity


class VarianceOfTokenProbs:
    """
    Variance of token probabilities across the sequence.

    Measures how consistent the model's confidence is throughout
    the generation. Higher variance indicates varying uncertainty.
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute variance of max token probabilities.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            mask: Optional mask for padding of shape (batch, seq_len)
            dim: Dimension to compute softmax over (default: -1)

        Returns:
            Variance of shape (batch,)
        """
        probs = F.softmax(logits, dim=dim)
        max_probs, _ = probs.max(dim=dim)

        if mask is not None:
            max_probs = max_probs * mask
            seq_lengths = mask.sum(dim=1)
            mean = max_probs.sum(dim=1) / seq_lengths

            # Variance
            diff = (max_probs - mean.unsqueeze(1)) ** 2
            diff = diff * mask
            variance = diff.sum(dim=1) / seq_lengths
        else:
            variance = max_probs.var(dim=1)

        return variance
