"""
Token-level uncertainty quantification for LLMs.

These methods measure uncertainty at each individual token position
during generation. They operate on token probability distributions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TokenEntropy:
    """
    Compute predictive entropy at each token position.

    Higher entropy indicates higher uncertainty in the model's prediction
    for that token. This is the most common token-level uncertainty measure.
    """

    @staticmethod
    def compute(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute entropy over token distribution.

        Args:
            logits: Token logits of shape (..., vocab_size)
            dim: Dimension to compute entropy over (default: -1)

        Returns:
            Entropy values of shape (...,)
        """
        probs = F.softmax(logits, dim=dim)
        log_probs = F.log_softmax(logits, dim=dim)
        entropy = -(probs * log_probs).sum(dim=dim)
        return entropy


class TokenConfidence:
    """
    Maximum softmax probability at each token position.

    The probability assigned to the most likely token. Lower values
    indicate higher uncertainty.
    """

    @staticmethod
    def compute(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute max softmax probability.

        Args:
            logits: Token logits of shape (..., vocab_size)
            dim: Dimension to compute over (default: -1)

        Returns:
            Max probabilities of shape (...,)
        """
        probs = F.softmax(logits, dim=dim)
        max_probs, _ = probs.max(dim=dim)
        return max_probs


class TokenPerplexity:
    """
    Perplexity at each token position.

    Perplexity = exp(entropy). It measures how "surprised" the model
    is by its own prediction. Higher values indicate more uncertainty.
    """

    @staticmethod
    def compute(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute perplexity per token.

        Args:
            logits: Token logits of shape (..., vocab_size)
            dim: Dimension to compute over (default: -1)

        Returns:
            Perplexity values of shape (...,)
        """
        entropy = TokenEntropy.compute(logits, dim=dim)
        perplexity = torch.exp(entropy)
        return perplexity


class SurprisalScore:
    """
    Surprisal (negative log-probability) of generated tokens.

    Also called information content. Higher values indicate the token
    was less expected by the model (higher uncertainty).
    """

    @staticmethod
    def compute(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute surprisal for specific tokens.

        Args:
            logits: Token logits of shape (batch, seq_len, vocab_size)
            token_ids: Generated token IDs of shape (batch, seq_len)
            dim: Dimension to compute over (default: -1)

        Returns:
            Surprisal values of shape (batch, seq_len)
        """
        log_probs = F.log_softmax(logits, dim=dim)

        # Gather log probs for the actual tokens
        # log_probs: (batch, seq_len, vocab_size)
        # token_ids: (batch, seq_len) -> (batch, seq_len, 1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

        # Surprisal is negative log probability
        surprisal = -token_log_probs
        return surprisal


class TopKConfidence:
    """
    Confidence based on probability mass in top-k tokens.

    Measures what fraction of probability mass is concentrated
    in the top k most likely tokens. Higher values indicate
    lower uncertainty (more concentrated distribution).
    """

    @staticmethod
    def compute(logits: torch.Tensor, k: int = 5, dim: int = -1) -> torch.Tensor:
        """
        Compute top-k probability mass.

        Args:
            logits: Token logits of shape (..., vocab_size)
            k: Number of top tokens to consider
            dim: Dimension to compute over (default: -1)

        Returns:
            Top-k probability mass of shape (...,)
        """
        probs = F.softmax(logits, dim=dim)
        top_k_probs, _ = probs.topk(k, dim=dim)
        top_k_mass = top_k_probs.sum(dim=dim)
        return top_k_mass
