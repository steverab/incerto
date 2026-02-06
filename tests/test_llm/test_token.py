"""
Tests for LLM token-level uncertainty methods.
"""

import torch

from incerto.llm import (
    TokenEntropy,
    TokenConfidence,
    TokenPerplexity,
    SurprisalScore,
    TopKConfidence,
)


class TestTokenEntropy:
    """Test token-level entropy computation."""

    def test_compute_shape(self, token_logits):
        """Test entropy has correct shape."""
        entropy = TokenEntropy.compute(token_logits)

        # Should return entropy per token (batch, seq_len)
        assert entropy.shape == token_logits.shape[:2]
        assert (entropy >= 0).all()  # Entropy is non-negative

    def test_uniform_distribution(self):
        """Test entropy is maximum for uniform distribution."""
        vocab_size = 100
        logits = torch.zeros(1, 10, vocab_size)  # Uniform

        entropy = TokenEntropy.compute(logits)

        # Should be close to log(vocab_size)
        max_entropy = torch.log(torch.tensor(float(vocab_size)))
        assert torch.allclose(entropy, max_entropy.expand_as(entropy), atol=0.1)

    def test_deterministic_distribution(self):
        """Test entropy is zero for deterministic distribution."""
        logits = torch.zeros(1, 10, 100)
        logits[:, :, 0] = 100.0  # Very peaked

        entropy = TokenEntropy.compute(logits)

        # Should be close to 0
        assert (entropy < 0.1).all()


class TestTokenConfidence:
    """Test token-level confidence (max probability)."""

    def test_compute_shape(self, token_logits):
        """Test confidence has correct shape."""
        confidence = TokenConfidence.compute(token_logits)

        assert confidence.shape == token_logits.shape[:2]
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_deterministic_high_confidence(self):
        """Test high confidence for peaked distributions."""
        logits = torch.zeros(1, 10, 100)
        logits[:, :, 0] = 100.0

        confidence = TokenConfidence.compute(logits)

        # Should be close to 1.0
        assert (confidence > 0.99).all()

    def test_uniform_low_confidence(self):
        """Test low confidence for uniform distributions."""
        vocab_size = 100
        logits = torch.zeros(1, 10, vocab_size)

        confidence = TokenConfidence.compute(logits)

        # Should be close to 1/vocab_size
        expected = 1.0 / vocab_size
        assert torch.allclose(confidence, torch.tensor(expected), atol=0.01)


class TestSurprisalScore:
    """Test surprisal scores."""

    def test_compute_shape(self, token_logits, token_ids):
        """Test surprisal has correct shape."""
        surprisal = SurprisalScore.compute(token_logits, token_ids)

        assert surprisal.shape == token_ids.shape
        assert (surprisal >= 0).all()  # Surprisal is non-negative

    def test_high_probability_low_surprisal(self):
        """Test high probability tokens have low surprisal."""
        logits = torch.zeros(1, 10, 100)
        token_ids = torch.zeros(1, 10, dtype=torch.long)

        # Make token 0 very likely
        logits[:, :, 0] = 100.0

        surprisal = SurprisalScore.compute(logits, token_ids)

        # Should have low surprisal
        assert (surprisal < 0.1).all()


class TestTokenPerplexity:
    """Test token perplexity."""

    def test_compute(self, token_logits):
        """Test perplexity computation."""
        perplexity = TokenPerplexity.compute(token_logits, dim=-1)

        assert (perplexity >= 1.0).all()  # Perplexity is >= 1
        assert torch.isfinite(perplexity).all()

    def test_perfect_predictions_low_perplexity(self):
        """Test perfect predictions have low perplexity."""
        logits = torch.zeros(1, 10, 100)

        # Make one token very likely (peaked distribution = low entropy = low perplexity)
        logits[:, :, 0] = 100.0

        perplexity = TokenPerplexity.compute(logits, dim=-1)

        # Perfect predictions should have perplexity close to 1.0
        assert (perplexity < 1.1).all()


class TestTopKConfidence:
    """Test top-k confidence."""

    def test_compute_shape(self, token_logits):
        """Test top-k confidence has correct shape."""
        confidence = TopKConfidence.compute(token_logits, k=5)

        assert confidence.shape == token_logits.shape[:2]
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_different_k(self, token_logits):
        """Test different values of k."""
        for k in [1, 5, 10, 50]:
            confidence = TopKConfidence.compute(token_logits, k=k)
            assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_k1_equals_max_prob(self, token_logits):
        """Test k=1 equals max probability."""
        conf_k1 = TopKConfidence.compute(token_logits, k=1)
        max_prob = TokenConfidence.compute(token_logits)

        assert torch.allclose(conf_k1, max_prob, atol=1e-5)
