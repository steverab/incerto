"""
Tests for LLM sampling-based uncertainty methods.
"""

import pytest
import torch

from incerto.llm import (
    SelfConsistency,
    LexicalSimilarity,
    VarianceRatio,
    PredictiveEntropy,
    MutualInformation,
    SemanticEntropy,
    EnsembleDisagreement,
)


class TestSelfConsistency:
    """Test self-consistency computation."""

    def test_compute_all_same(self):
        """Test all identical responses gives perfect agreement."""
        responses = ["Paris", "Paris", "Paris", "Paris", "Paris"]
        result = SelfConsistency.compute(responses)

        assert result["agreement_rate"] == 1.0
        assert result["entropy"] == pytest.approx(0.0, abs=1e-9)
        assert result["top_response"] == "Paris"
        assert result["num_unique"] == 1
        assert result["confidence"] == 1.0

    def test_compute_all_different(self):
        """Test all different responses gives low agreement."""
        responses = ["Paris", "London", "Berlin", "Tokyo", "Madrid"]
        result = SelfConsistency.compute(responses)

        assert result["agreement_rate"] == 0.2  # 1/5
        assert result["num_unique"] == 5
        assert result["entropy"] > 0

    def test_compute_mixed(self):
        """Test mixed responses."""
        responses = ["Paris", "Paris", "Paris", "London", "Berlin"]
        result = SelfConsistency.compute(responses)

        assert result["agreement_rate"] == 0.6  # 3/5
        assert result["top_response"] == "Paris"
        assert result["num_unique"] == 3

    def test_with_normalize_fn(self):
        """Test with normalization function."""
        responses = ["Paris", "paris", "PARIS", "Paris.", "paris!"]

        def normalize(x):
            return x.lower().strip("!.")

        result = SelfConsistency.compute(responses, normalize_fn=normalize)

        assert result["agreement_rate"] == 1.0
        assert result["num_unique"] == 1


class TestLexicalSimilarity:
    """Test lexical similarity methods."""

    def test_exact_match_rate_all_same(self):
        """Test exact match rate with all identical."""
        responses = ["The answer is 42", "The answer is 42", "The answer is 42"]
        rate = LexicalSimilarity.exact_match_rate(responses)
        assert rate == 1.0

    def test_exact_match_rate_all_different(self):
        """Test exact match rate with all different."""
        responses = ["Answer A", "Answer B", "Answer C"]
        rate = LexicalSimilarity.exact_match_rate(responses)
        assert rate == pytest.approx(1 / 3)

    def test_pairwise_token_overlap_identical(self):
        """Test Jaccard similarity for identical responses."""
        responses = ["hello world", "hello world", "hello world"]
        overlap = LexicalSimilarity.pairwise_token_overlap(responses)
        assert overlap == 1.0

    def test_pairwise_token_overlap_no_overlap(self):
        """Test Jaccard similarity with no token overlap."""
        responses = ["apple banana", "cat dog"]
        overlap = LexicalSimilarity.pairwise_token_overlap(responses)
        assert overlap == 0.0

    def test_pairwise_token_overlap_partial(self):
        """Test Jaccard similarity with partial overlap."""
        responses = ["the quick brown", "the slow brown"]
        # Intersection: {the, brown}, Union: {the, quick, slow, brown}
        # Jaccard = 2/4 = 0.5
        overlap = LexicalSimilarity.pairwise_token_overlap(responses)
        assert overlap == pytest.approx(0.5)

    def test_single_response(self):
        """Test with single response returns 1.0."""
        responses = ["single response"]
        overlap = LexicalSimilarity.pairwise_token_overlap(responses)
        assert overlap == 1.0


class TestVarianceRatio:
    """Test variance ratio for classification."""

    def test_all_same_prediction(self):
        """Test all same predictions gives zero variance ratio."""
        predictions = [0, 0, 0, 0, 0]
        vr = VarianceRatio.compute(predictions)
        assert vr == 0.0

    def test_all_different_predictions(self):
        """Test all different predictions gives high variance ratio."""
        predictions = [0, 1, 2, 3, 4]
        vr = VarianceRatio.compute(predictions)
        assert vr == 0.8  # 1 - 1/5

    def test_mixed_predictions(self):
        """Test mixed predictions."""
        predictions = [0, 0, 0, 1, 1]  # 3 zeros, 2 ones
        vr = VarianceRatio.compute(predictions)
        assert vr == pytest.approx(0.4)  # 1 - 3/5


class TestPredictiveEntropy:
    """Test predictive entropy computation."""

    def test_compute_shape(self):
        """Test output shape matches input sequence length."""
        logits1 = torch.randn(10, 100)  # seq_len=10, vocab=100
        logits2 = torch.randn(10, 100)
        logits3 = torch.randn(10, 100)

        entropy = PredictiveEntropy.compute([logits1, logits2, logits3])
        assert entropy.shape == (10,)

    def test_entropy_non_negative(self):
        """Test entropy is non-negative."""
        logits = [torch.randn(5, 50) for _ in range(3)]
        entropy = PredictiveEntropy.compute(logits)
        assert (entropy >= 0).all()

    def test_single_sample(self):
        """Test with single sample."""
        logits = [torch.randn(5, 50)]
        entropy = PredictiveEntropy.compute(logits)
        assert entropy.shape == (5,)


class TestMutualInformation:
    """Test mutual information computation."""

    def test_compute_shape(self):
        """Test output shape."""
        logits = [torch.randn(10, 100) for _ in range(5)]
        mi = MutualInformation.compute(logits)
        assert mi.shape == (10,)

    def test_mi_non_negative(self):
        """Test MI is non-negative."""
        logits = [torch.randn(5, 50) for _ in range(3)]
        mi = MutualInformation.compute(logits)
        # MI should be non-negative (with small tolerance for numerical errors)
        assert (mi >= -1e-5).all()

    def test_identical_samples_zero_mi(self):
        """Test identical samples have zero MI."""
        base_logits = torch.randn(5, 50)
        logits = [base_logits.clone() for _ in range(5)]

        mi = MutualInformation.compute(logits)
        assert torch.allclose(mi, torch.zeros_like(mi), atol=1e-5)


class TestSemanticEntropy:
    """Test semantic entropy computation."""

    def test_compute_empty(self):
        """Test with empty responses."""
        result = SemanticEntropy.compute([])
        assert result["semantic_entropy"] == 0.0
        assert result["num_clusters"] == 0
        assert result["clusters"] == []

    def test_compute_single(self):
        """Test with single response."""
        result = SemanticEntropy.compute(["Hello world"])
        assert result["semantic_entropy"] == 0.0
        assert result["num_clusters"] == 1
        assert result["clusters"] == [0]

    def test_compute_all_same(self):
        """Test all identical responses."""
        responses = ["Paris", "Paris", "Paris"]
        result = SemanticEntropy.compute(responses, similarity_threshold=0.5)
        # All should be in same cluster
        assert result["num_clusters"] == 1
        assert result["semantic_entropy"] == pytest.approx(0.0, abs=1e-9)

    def test_compute_all_different(self):
        """Test all different responses with low threshold."""
        responses = ["apple", "banana", "cherry"]
        result = SemanticEntropy.compute(responses, similarity_threshold=0.99)
        # Each should be its own cluster with high threshold
        assert result["num_clusters"] == 3

    def test_cluster_assignments(self):
        """Test cluster assignments are valid."""
        responses = ["cat dog", "cat dog", "fish bird", "fish bird"]
        result = SemanticEntropy.compute(responses, similarity_threshold=0.5)

        clusters = result["clusters"]
        assert len(clusters) == 4
        assert all(c >= 0 for c in clusters)
        # First two should be in same cluster, last two in another
        assert clusters[0] == clusters[1]
        assert clusters[2] == clusters[3]


class TestEnsembleDisagreement:
    """Test ensemble disagreement computation."""

    def test_all_agree(self):
        """Test all models agree gives zero disagreement."""
        # 3 models, 5 samples, all predict class 0
        predictions = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        disagreement = EnsembleDisagreement.compute(predictions)
        assert disagreement == 0.0

    def test_all_disagree(self):
        """Test all models disagree gives full disagreement."""
        # 3 models, 5 samples, all disagree on every sample
        predictions = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]]
        disagreement = EnsembleDisagreement.compute(predictions)
        assert disagreement == 1.0

    def test_partial_disagreement(self):
        """Test partial disagreement."""
        # 2 models, 4 samples: agree on 2, disagree on 2
        predictions = [[0, 0, 1, 1], [0, 0, 2, 2]]
        disagreement = EnsembleDisagreement.compute(predictions)
        assert disagreement == 0.5

    def test_single_model(self):
        """Test single model gives zero disagreement."""
        predictions = [[0, 1, 2, 3]]
        disagreement = EnsembleDisagreement.compute(predictions)
        assert disagreement == 0.0
