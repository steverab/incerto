"""
Tests for OOD detection metrics.
"""

import pytest
import torch

from incerto.ood.metrics import auroc, detection_accuracy, fpr_at_tpr


class TestAUROC:
    """Test AUROC computation."""

    def test_perfect_separation(self):
        """Perfect separation should give AUROC = 1.0."""
        id_scores = torch.zeros(50)
        ood_scores = torch.ones(50)
        assert auroc(id_scores, ood_scores) == pytest.approx(1.0)

    def test_random_scores(self, set_seed):
        """Random scores should give AUROC ~ 0.5."""
        id_scores = torch.randn(200)
        ood_scores = torch.randn(200)
        result = auroc(id_scores, ood_scores)
        assert 0.3 < result < 0.7

    def test_inverted_separation(self):
        """Inverted separation should give AUROC = 0.0."""
        id_scores = torch.ones(50)
        ood_scores = torch.zeros(50)
        assert auroc(id_scores, ood_scores) == pytest.approx(0.0)

    def test_return_type(self):
        """Should return a float."""
        id_scores = torch.randn(20)
        ood_scores = torch.randn(20)
        result = auroc(id_scores, ood_scores)
        assert isinstance(result, float)

    def test_different_sizes(self):
        """Should handle different-sized ID and OOD score arrays."""
        id_scores = torch.zeros(30)
        ood_scores = torch.ones(70)
        assert auroc(id_scores, ood_scores) == pytest.approx(1.0)


class TestFPRAtTPR:
    """Test FPR at target TPR."""

    def test_perfect_separation(self):
        """Perfect separation should give FPR = 0.0 at any TPR."""
        id_scores = torch.zeros(100)
        ood_scores = torch.ones(100)
        result = fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
        assert result == pytest.approx(0.0, abs=0.02)

    def test_random_scores(self, set_seed):
        """Random scores at 95% TPR should give FPR ~ 0.95."""
        id_scores = torch.randn(500)
        ood_scores = torch.randn(500)
        result = fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
        assert 0.8 < result < 1.0

    def test_return_type(self):
        """Should return a float."""
        id_scores = torch.randn(50)
        ood_scores = torch.randn(50)
        result = fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
        assert isinstance(result, float)

    def test_fpr_range(self):
        """FPR should be in [0, 1]."""
        id_scores = torch.randn(100)
        ood_scores = torch.randn(100) + 1.0
        result = fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
        assert 0.0 <= result <= 1.0

    def test_different_tpr_targets(self):
        """Higher TPR target should give higher FPR."""
        id_scores = torch.randn(200)
        ood_scores = torch.randn(200) + 1.0
        fpr_90 = fpr_at_tpr(id_scores, ood_scores, tpr=0.90)
        fpr_99 = fpr_at_tpr(id_scores, ood_scores, tpr=0.99)
        assert fpr_99 >= fpr_90


class TestDetectionAccuracy:
    """Test detection accuracy computation."""

    def test_perfect_separation(self):
        """Perfect separation should give high accuracy."""
        id_scores = torch.zeros(100)
        ood_scores = torch.ones(100)
        acc = detection_accuracy(id_scores, ood_scores)
        assert acc > 0.9

    def test_return_type(self):
        """Should return a float."""
        id_scores = torch.randn(50)
        ood_scores = torch.randn(50)
        result = detection_accuracy(id_scores, ood_scores)
        assert isinstance(result, float)

    def test_accuracy_range(self):
        """Accuracy should be in [0, 1]."""
        id_scores = torch.randn(100)
        ood_scores = torch.randn(100)
        result = detection_accuracy(id_scores, ood_scores)
        assert 0.0 <= result <= 1.0

    def test_custom_accept_rate(self):
        """Different accept rates should produce different thresholds."""
        id_scores = torch.randn(200)
        ood_scores = torch.randn(200) + 2.0
        acc_90 = detection_accuracy(id_scores, ood_scores, id_accept_rate=0.90)
        acc_99 = detection_accuracy(id_scores, ood_scores, id_accept_rate=0.99)
        # Higher accept rate → higher threshold → fewer OOD detected
        assert acc_90 != acc_99

    def test_default_accept_rate(self):
        """Default accept rate of 0.95 should work."""
        id_scores = torch.zeros(100)
        ood_scores = torch.ones(100)
        acc_default = detection_accuracy(id_scores, ood_scores)
        acc_explicit = detection_accuracy(id_scores, ood_scores, id_accept_rate=0.95)
        assert acc_default == acc_explicit
