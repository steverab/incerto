"""
Tests for LLM generation-specific uncertainty methods.
"""

import pytest
import torch

from incerto.llm import (
    BeamSearchUncertainty,
    NucleusSamplingUncertainty,
    IDontKnowDetection,
    ContrastiveDecoding,
)


class TestBeamSearchUncertainty:
    """Test beam search uncertainty computation."""

    def test_compute_from_scores_shape(self):
        """Test result dictionary has expected keys."""
        beam_scores = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        result = BeamSearchUncertainty.compute_from_scores(beam_scores)

        assert "entropy" in result
        assert "top_beam_prob" in result
        assert "score_variance" in result
        assert "confidence" in result

    def test_compute_from_scores_values(self):
        """Test computed values are reasonable."""
        beam_scores = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        result = BeamSearchUncertainty.compute_from_scores(beam_scores)

        assert result["entropy"] >= 0
        assert 0 <= result["top_beam_prob"] <= 1
        assert result["score_variance"] >= 0
        assert result["confidence"] == result["top_beam_prob"]

    def test_single_dominant_beam(self):
        """Test with one dominant beam score."""
        # First beam much better than others
        beam_scores = torch.tensor([0.0, -100.0, -100.0, -100.0, -100.0])
        result = BeamSearchUncertainty.compute_from_scores(beam_scores)

        # Should have low entropy, high top_beam_prob
        assert result["entropy"] < 0.1
        assert result["top_beam_prob"] > 0.99

    def test_temperature_effect(self):
        """Test temperature affects entropy."""
        beam_scores = torch.tensor([-1.0, -2.0, -3.0])

        result_t1 = BeamSearchUncertainty.compute_from_scores(
            beam_scores, temperature=1.0
        )
        result_t01 = BeamSearchUncertainty.compute_from_scores(
            beam_scores, temperature=0.1
        )

        # Lower temperature should sharpen distribution -> lower entropy
        assert result_t01["entropy"] < result_t1["entropy"]

    def test_diversity_among_beams_identical(self):
        """Test diversity with identical sequences."""
        sequences = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        diversity = BeamSearchUncertainty.diversity_among_beams(sequences)
        assert diversity == pytest.approx(1 / 3)

    def test_diversity_among_beams_all_different(self):
        """Test diversity with all different sequences."""
        sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        diversity = BeamSearchUncertainty.diversity_among_beams(sequences)
        assert diversity == 1.0

    def test_diversity_single_beam(self):
        """Test diversity with single beam."""
        sequences = [[1, 2, 3]]
        diversity = BeamSearchUncertainty.diversity_among_beams(sequences)
        assert diversity == 0.0


class TestNucleusSamplingUncertainty:
    """Test nucleus sampling uncertainty methods."""

    def test_effective_vocabulary_size(self):
        """Test effective vocabulary size computation."""
        vocab_size = 100
        logits = torch.zeros(vocab_size)
        # Make first 10 tokens have most probability
        logits[:10] = 10.0

        eff_size = NucleusSamplingUncertainty.effective_vocabulary_size(logits, p=0.9)
        # Should be close to 10 (where most mass is)
        assert eff_size <= 15

    def test_effective_vocabulary_size_uniform(self):
        """Test effective vocab for uniform distribution."""
        vocab_size = 100
        logits = torch.zeros(vocab_size)

        eff_size = NucleusSamplingUncertainty.effective_vocabulary_size(logits, p=0.9)
        # Need 91 tokens to cover >= 90% of uniform distribution (cumsum <= p gives 90, +1 = 91)
        assert eff_size == 91

    def test_probability_mass_concentration(self):
        """Test probability mass concentration in top-k."""
        vocab_size = 100
        logits = torch.zeros(vocab_size)
        logits[0] = 100.0  # First token dominates

        mass = NucleusSamplingUncertainty.probability_mass_concentration(
            logits, top_k=1
        )
        assert mass > 0.99

    def test_probability_mass_concentration_uniform(self):
        """Test probability mass for uniform distribution."""
        vocab_size = 100
        logits = torch.zeros(vocab_size)

        mass = NucleusSamplingUncertainty.probability_mass_concentration(
            logits, top_k=10
        )
        # Top-10 of uniform 100 should have ~10% mass
        assert mass == pytest.approx(0.1, abs=0.01)


class TestIDontKnowDetection:
    """Test uncertainty phrase detection."""

    def test_contains_uncertainty_phrase_positive(self):
        """Test detection of uncertainty phrases."""
        texts = [
            "I don't know the answer",
            "I'm not sure about this",
            "This is unclear to me",
            "I cannot determine the result",
            "There's not enough information",
        ]
        for text in texts:
            assert IDontKnowDetection.contains_uncertainty_phrase(text)

    def test_contains_uncertainty_phrase_negative(self):
        """Test absence of uncertainty phrases."""
        texts = [
            "The answer is 42",
            "Paris is the capital of France",
            "The result is definitely correct",
        ]
        for text in texts:
            assert not IDontKnowDetection.contains_uncertainty_phrase(text)

    def test_case_insensitive(self):
        """Test detection is case-insensitive."""
        assert IDontKnowDetection.contains_uncertainty_phrase("I DON'T KNOW")
        assert IDontKnowDetection.contains_uncertainty_phrase("i don't know")
        assert IDontKnowDetection.contains_uncertainty_phrase("I Don't Know")

    def test_extract_hedging_no_hedges(self):
        """Test hedging extraction with no hedging."""
        result = IDontKnowDetection.extract_confidence_from_hedging(
            "The answer is definitely 42"
        )
        assert result["hedges_found"] == []
        assert result["num_hedges"] == 0
        assert result["estimated_confidence"] == 1.0
        assert not result["contains_hedging"]

    def test_extract_hedging_with_hedges(self):
        """Test hedging extraction with hedging language."""
        result = IDontKnowDetection.extract_confidence_from_hedging(
            "Maybe the answer is probably around 42"
        )
        assert "maybe" in result["hedges_found"]
        assert "probably" in result["hedges_found"]
        assert result["num_hedges"] == 2
        assert result["estimated_confidence"] < 1.0
        assert result["contains_hedging"]

    def test_extract_hedging_single_hedge(self):
        """Test hedging extraction with single hedge word."""
        result = IDontKnowDetection.extract_confidence_from_hedging(
            "It seems like the answer is 42"
        )
        assert "seems" in result["hedges_found"]
        assert result["estimated_confidence"] == 0.6


class TestContrastiveDecoding:
    """Test contrastive decoding methods."""

    def test_compute_contrastive_score_shape(self):
        """Test contrastive score has correct shape."""
        expert_logits = torch.randn(10, 100)
        amateur_logits = torch.randn(10, 100)

        scores = ContrastiveDecoding.compute_contrastive_score(
            expert_logits, amateur_logits
        )
        assert scores.shape == (10, 100)

    def test_compute_contrastive_score_alpha_effect(self):
        """Test alpha parameter affects scores."""
        expert_logits = torch.randn(5, 50)
        amateur_logits = torch.randn(5, 50)

        scores_alpha05 = ContrastiveDecoding.compute_contrastive_score(
            expert_logits, amateur_logits, alpha=0.5
        )
        scores_alpha10 = ContrastiveDecoding.compute_contrastive_score(
            expert_logits, amateur_logits, alpha=1.0
        )

        # Different alpha should give different scores
        assert not torch.allclose(scores_alpha05, scores_alpha10)

    def test_contrastive_score_identical_models(self):
        """Test contrastive score when models are identical."""
        logits = torch.randn(5, 50)

        scores = ContrastiveDecoding.compute_contrastive_score(
            logits, logits, alpha=1.0
        )
        # When expert == amateur, score = expert_prob - 1*amateur_prob = 0
        assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-5)

    def test_disagreement_score_shape(self):
        """Test disagreement score has correct shape."""
        expert_logits = torch.randn(10, 100)
        amateur_logits = torch.randn(10, 100)

        disagreement = ContrastiveDecoding.disagreement_score(
            expert_logits, amateur_logits
        )
        assert disagreement.shape == (10,)

    def test_disagreement_score_non_negative(self):
        """Test disagreement score is non-negative (KL divergence)."""
        expert_logits = torch.randn(5, 50)
        amateur_logits = torch.randn(5, 50)

        disagreement = ContrastiveDecoding.disagreement_score(
            expert_logits, amateur_logits
        )
        assert (disagreement >= -1e-5).all()

    def test_disagreement_score_identical_zero(self):
        """Test disagreement is zero for identical distributions."""
        logits = torch.randn(5, 50)

        disagreement = ContrastiveDecoding.disagreement_score(logits, logits)
        assert torch.allclose(disagreement, torch.zeros_like(disagreement), atol=1e-5)
