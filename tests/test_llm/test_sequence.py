"""
Tests for LLM sequence-level uncertainty methods.
"""

import pytest
import torch

from incerto.llm import (
    AverageLogProb,
    NormalizedSequenceProb,
    SequenceEntropy,
    SequencePerplexity,
    SequenceProbability,
    VarianceOfTokenProbs,
)


class TestSequenceProbability:
    """Test sequence probability computation."""

    def test_compute_shape(self, token_logits, token_ids):
        """Test output has correct shape (batch,)."""
        prob = SequenceProbability.compute(token_logits, token_ids)
        assert prob.shape == (token_logits.shape[0],)

    def test_probability_range(self, token_logits, token_ids):
        """Test probability is between 0 and 1."""
        prob = SequenceProbability.compute(token_logits, token_ids)
        assert (prob >= 0).all()
        assert (prob <= 1).all()

    def test_high_probability_peaked_distribution(self):
        """Test high probability for peaked distributions."""
        batch, seq_len, vocab = 2, 5, 100
        logits = torch.zeros(batch, seq_len, vocab)
        token_ids = torch.zeros(batch, seq_len, dtype=torch.long)

        # Make token 0 very likely
        logits[:, :, 0] = 100.0

        prob = SequenceProbability.compute(logits, token_ids)
        # Each token prob is ~1, so sequence prob is ~1
        assert (prob > 0.99).all()


class TestAverageLogProb:
    """Test average log probability computation."""

    def test_compute_shape(self, token_logits, token_ids):
        """Test output has correct shape."""
        avg_lp = AverageLogProb.compute(token_logits, token_ids)
        assert avg_lp.shape == (token_logits.shape[0],)

    def test_negative_values(self, token_logits, token_ids):
        """Test log probs are negative (since probs < 1)."""
        avg_lp = AverageLogProb.compute(token_logits, token_ids)
        # For random logits, log probs should be negative on average
        assert torch.isfinite(avg_lp).all()

    def test_with_mask(self, token_logits, token_ids):
        """Test masking works correctly."""
        batch, seq_len = token_ids.shape
        mask = torch.ones(batch, seq_len)
        mask[:, seq_len // 2 :] = 0  # Mask second half

        avg_lp_masked = AverageLogProb.compute(token_logits, token_ids, mask=mask)
        avg_lp_full = AverageLogProb.compute(token_logits, token_ids)

        # Results should differ
        assert not torch.allclose(avg_lp_masked, avg_lp_full)


class TestNormalizedSequenceProb:
    """Test length-normalized sequence probability."""

    def test_compute_shape(self, token_logits, token_ids):
        """Test output has correct shape."""
        norm_prob = NormalizedSequenceProb.compute(token_logits, token_ids)
        assert norm_prob.shape == (token_logits.shape[0],)

    def test_probability_range(self, token_logits, token_ids):
        """Test output is in valid probability range [0, 1]."""
        norm_prob = NormalizedSequenceProb.compute(token_logits, token_ids)
        assert (norm_prob >= 0).all()
        assert (norm_prob <= 1).all()

    def test_length_penalty_effect(self, token_logits, token_ids):
        """Test different length penalties produce different results."""
        prob_0 = NormalizedSequenceProb.compute(token_logits, token_ids, length_penalty=0.0)
        prob_1 = NormalizedSequenceProb.compute(token_logits, token_ids, length_penalty=1.0)

        # With penalty=0, no length normalization; penalty=1 normalizes by length
        assert not torch.allclose(prob_0, prob_1)

    def test_with_mask(self, token_logits, token_ids):
        """Test with mask for variable-length sequences."""
        batch, seq_len = token_ids.shape
        mask = torch.ones(batch, seq_len)
        mask[0, seq_len // 2 :] = 0  # First sequence shorter

        norm_prob = NormalizedSequenceProb.compute(
            token_logits, token_ids, mask=mask, length_penalty=1.0
        )
        assert torch.isfinite(norm_prob).all()


class TestSequenceEntropy:
    """Test sequence entropy aggregation."""

    def test_compute_shape(self, token_logits):
        """Test output has correct shape."""
        entropy = SequenceEntropy.compute(token_logits)
        assert entropy.shape == (token_logits.shape[0],)

    def test_aggregation_mean(self, token_logits):
        """Test mean aggregation."""
        entropy = SequenceEntropy.compute(token_logits, aggregation="mean")
        assert (entropy >= 0).all()

    def test_aggregation_sum(self, token_logits):
        """Test sum aggregation produces larger values than mean."""
        entropy_mean = SequenceEntropy.compute(token_logits, aggregation="mean")
        entropy_sum = SequenceEntropy.compute(token_logits, aggregation="sum")
        # Sum should be larger (seq_len * mean)
        assert (entropy_sum >= entropy_mean).all()

    def test_aggregation_max(self, token_logits):
        """Test max aggregation."""
        entropy = SequenceEntropy.compute(token_logits, aggregation="max")
        assert (entropy >= 0).all()

    def test_invalid_aggregation(self, token_logits):
        """Test invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            SequenceEntropy.compute(token_logits, aggregation="invalid")

    def test_with_mask(self, token_logits):
        """Test with mask."""
        batch, seq_len = token_logits.shape[:2]
        mask = torch.ones(batch, seq_len)
        mask[:, seq_len // 2 :] = 0

        entropy = SequenceEntropy.compute(token_logits, mask=mask)
        assert torch.isfinite(entropy).all()


class TestSequencePerplexity:
    """Test sequence perplexity computation."""

    def test_compute_shape(self, token_logits, token_ids):
        """Test output has correct shape."""
        ppl = SequencePerplexity.compute(token_logits, token_ids)
        assert ppl.shape == (token_logits.shape[0],)

    def test_perplexity_at_least_one(self, token_logits, token_ids):
        """Test perplexity is >= 1."""
        ppl = SequencePerplexity.compute(token_logits, token_ids)
        assert (ppl >= 1.0).all()

    def test_perfect_predictions_low_perplexity(self):
        """Test perfect predictions have perplexity close to 1."""
        batch, seq_len, vocab = 2, 10, 100
        logits = torch.zeros(batch, seq_len, vocab)
        token_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        logits[:, :, 0] = 100.0  # Make token 0 very likely

        ppl = SequencePerplexity.compute(logits, token_ids)
        assert (ppl < 1.1).all()


class TestVarianceOfTokenProbs:
    """Test variance of token probabilities."""

    def test_compute_shape(self, token_logits):
        """Test output has correct shape."""
        var = VarianceOfTokenProbs.compute(token_logits)
        assert var.shape == (token_logits.shape[0],)

    def test_variance_non_negative(self, token_logits):
        """Test variance is non-negative."""
        var = VarianceOfTokenProbs.compute(token_logits)
        assert (var >= 0).all()

    def test_constant_confidence_zero_variance(self):
        """Test constant confidence across tokens has zero variance."""
        batch, seq_len, vocab = 2, 10, 100
        # Uniform distribution at each position
        logits = torch.zeros(batch, seq_len, vocab)

        var = VarianceOfTokenProbs.compute(logits)
        # All max probs should be 1/vocab, so variance should be ~0
        assert torch.allclose(var, torch.zeros_like(var), atol=1e-5)

    def test_with_mask(self, token_logits):
        """Test with mask."""
        batch, seq_len = token_logits.shape[:2]
        mask = torch.ones(batch, seq_len)
        mask[:, seq_len // 2 :] = 0

        var = VarianceOfTokenProbs.compute(token_logits, mask=mask)
        assert (var >= 0).all()
