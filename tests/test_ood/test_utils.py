"""
Tests for OOD detection utilities.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from incerto.ood.utils import (
    compute_threshold_at_tpr,
    get_ood_predictions,
    extract_features,
)


class TestComputeThresholdAtTPR:
    """Test threshold computation for OOD detection."""

    def test_basic_threshold(self):
        """Test threshold at 95% TPR."""
        id_scores = torch.linspace(0, 1, 100)
        threshold = compute_threshold_at_tpr(id_scores, target_tpr=0.95)
        assert threshold == pytest.approx(0.95, abs=0.02)

    def test_numpy_input(self):
        """Should accept numpy arrays."""
        id_scores = np.linspace(0, 1, 100)
        threshold = compute_threshold_at_tpr(id_scores, target_tpr=0.90)
        assert threshold == pytest.approx(0.90, abs=0.02)

    def test_return_type(self):
        """Should return a float."""
        id_scores = torch.randn(50)
        threshold = compute_threshold_at_tpr(id_scores, target_tpr=0.95)
        assert isinstance(threshold, float)

    def test_different_tpr_values(self):
        """Higher TPR should give higher threshold."""
        id_scores = torch.linspace(0, 10, 1000)
        thresh_50 = compute_threshold_at_tpr(id_scores, target_tpr=0.50)
        thresh_90 = compute_threshold_at_tpr(id_scores, target_tpr=0.90)
        thresh_99 = compute_threshold_at_tpr(id_scores, target_tpr=0.99)
        assert thresh_50 < thresh_90 < thresh_99

    def test_constant_scores(self):
        """Constant scores should return that value."""
        id_scores = torch.ones(100) * 5.0
        threshold = compute_threshold_at_tpr(id_scores, target_tpr=0.95)
        assert threshold == pytest.approx(5.0)


class TestGetOODPredictions:
    """Test binary OOD predictions."""

    def test_basic_predictions(self):
        """Test basic threshold-based predictions."""
        scores = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        preds = get_ood_predictions(scores, threshold=0.5)
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(preds, expected)

    def test_numpy_input(self):
        """Should accept numpy arrays."""
        scores = np.array([0.1, 0.6, 0.9])
        preds = get_ood_predictions(scores, threshold=0.5)
        expected = np.array([0, 1, 1])
        np.testing.assert_array_equal(preds, expected)

    def test_return_type(self):
        """Should return numpy array of ints."""
        scores = torch.randn(20)
        preds = get_ood_predictions(scores, threshold=0.0)
        assert isinstance(preds, np.ndarray)
        assert preds.dtype == np.int64 or preds.dtype == np.int32

    def test_all_below_threshold(self):
        """All below threshold should give all zeros."""
        scores = torch.zeros(10)
        preds = get_ood_predictions(scores, threshold=1.0)
        assert preds.sum() == 0

    def test_all_above_threshold(self):
        """All above threshold should give all ones."""
        scores = torch.ones(10) * 5.0
        preds = get_ood_predictions(scores, threshold=1.0)
        assert preds.sum() == 10


class TestExtractFeatures:
    """Test feature extraction utility."""

    @pytest.fixture
    def test_model(self):
        """Create a simple model with named layers."""
        import torch.nn as nn

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                )
                self.penultimate = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                self.classifier = nn.Linear(16, 5)

            def forward(self, x):
                x = self.features(x)
                x = self.penultimate(x)
                return self.classifier(x)

        return TestModel()

    @pytest.fixture
    def test_loader(self):
        """Create a simple data loader."""
        X = torch.randn(50, 10)
        y = torch.zeros(50).long()
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=16)

    def test_basic_extraction(self, test_model, test_loader):
        """Test feature extraction from a layer."""
        features = extract_features(test_model, test_loader, layer_name="penultimate")
        assert features.shape == (50, 16)

    def test_different_layer(self, test_model, test_loader):
        """Test extraction from different layers."""
        feat_pen = extract_features(test_model, test_loader, layer_name="penultimate")
        feat_feat = extract_features(test_model, test_loader, layer_name="features")
        assert feat_pen.shape == (50, 16)
        assert feat_feat.shape == (50, 32)

    def test_invalid_layer_raises(self, test_model, test_loader):
        """Invalid layer name should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            extract_features(test_model, test_loader, layer_name="nonexistent_layer")

    def test_return_type(self, test_model, test_loader):
        """Should return a tensor."""
        features = extract_features(test_model, test_loader, layer_name="penultimate")
        assert isinstance(features, torch.Tensor)

    def test_hook_removed_after_extraction(self, test_model, test_loader):
        """Hook should not persist after extraction."""
        # Count hooks before
        hooks_before = sum(len(m._forward_hooks) for m in test_model.modules())

        extract_features(test_model, test_loader, layer_name="penultimate")

        # Count hooks after
        hooks_after = sum(len(m._forward_hooks) for m in test_model.modules())

        assert hooks_after == hooks_before

    def test_endswith_matching(self, test_model, test_loader):
        """Should use endswith matching like Mahalanobis/KNN."""
        # "penultimate" should match "penultimate" layer
        features = extract_features(test_model, test_loader, layer_name="penultimate")
        assert features.shape[0] == 50

    def test_empty_dataloader(self, test_model):
        """Empty dataloader should return empty tensor."""
        X = torch.randn(0, 10)
        y = torch.zeros(0).long()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        features = extract_features(test_model, loader, layer_name="penultimate")
        assert features.numel() == 0
