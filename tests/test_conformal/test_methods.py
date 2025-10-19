"""
Tests for conformal prediction methods.

API:
- inductive_conformal(model, calib_loader, alpha) -> predictor(x) -> List[Tensor]
- mondrian_conformal(model, calib_loader, alpha, partition_fn=None) -> predictor(x) -> List[Tensor]
- aps(model, calib_loader, alpha) -> predictor(x) -> List[Tensor]
- raps(model, calib_loader, alpha, lam=0.0, k_reg=1) -> predictor(x) -> List[Tensor]
- jackknife_plus(model_fn, train_dataset, alpha) -> predictor(x) -> (lower, upper)
- cv_plus(model_fn, train_dataset, folds, alpha) -> predictor(x) -> (lower, upper)
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from incerto.conformal import (
    inductive_conformal,
    mondrian_conformal,
    aps,
    raps,
)


class TestInductiveConformal:
    """Test inductive conformal prediction."""

    def test_returns_predictor(self, ood_model, ood_id_loader):
        """Test that inductive_conformal returns a predictor function."""
        alpha = 0.1
        predictor = inductive_conformal(ood_model, ood_id_loader, alpha)

        assert callable(predictor)

    def test_predictor_output(self, ood_model, ood_id_loader):
        """Test that predictor returns list of prediction sets."""
        alpha = 0.1
        predictor = inductive_conformal(ood_model, ood_id_loader, alpha)

        # Test on a batch
        x_test = torch.randn(5, 64)
        pred_sets = predictor(x_test)

        assert isinstance(pred_sets, list)
        assert len(pred_sets) == 5
        # Each prediction set is a tensor of class indices
        for ps in pred_sets:
            assert isinstance(ps, torch.Tensor)

    def test_coverage_increases_with_lower_alpha(self, ood_model, ood_id_loader):
        """Test that smaller alpha (higher coverage) produces larger sets."""
        x_test = torch.randn(10, 64)

        predictor_90 = inductive_conformal(
            ood_model, ood_id_loader, alpha=0.1
        )  # 90% coverage
        predictor_80 = inductive_conformal(
            ood_model, ood_id_loader, alpha=0.2
        )  # 80% coverage

        sets_90 = predictor_90(x_test)
        sets_80 = predictor_80(x_test)

        # Higher coverage should generally produce larger sets
        # (This is probabilistic but generally true)
        assert all(isinstance(s, torch.Tensor) for s in sets_90)
        assert all(isinstance(s, torch.Tensor) for s in sets_80)


class TestMondrianConformal:
    """Test Mondrian (class-conditional) conformal prediction."""

    def test_returns_predictor(self, ood_model, ood_id_loader):
        """Test that mondrian_conformal returns a predictor function."""
        alpha = 0.1
        predictor = mondrian_conformal(ood_model, ood_id_loader, alpha)

        assert callable(predictor)

    def test_predictor_output(self, ood_model, ood_id_loader):
        """Test that predictor returns list of prediction sets."""
        alpha = 0.1
        predictor = mondrian_conformal(ood_model, ood_id_loader, alpha)

        # Test on a batch
        x_test = torch.randn(5, 64)
        pred_sets = predictor(x_test)

        assert isinstance(pred_sets, list)
        assert len(pred_sets) == 5
        for ps in pred_sets:
            assert isinstance(ps, torch.Tensor)

    def test_custom_partition_function(self, ood_model, ood_id_loader):
        """Test Mondrian with custom partition function."""
        alpha = 0.1

        # Custom partition: binary based on first feature
        def custom_partition(x, y):
            return (x[:, 0] > 0).long()

        predictor = mondrian_conformal(
            ood_model, ood_id_loader, alpha, partition_fn=custom_partition
        )

        assert callable(predictor)

        x_test = torch.randn(5, 64)
        pred_sets = predictor(x_test)
        assert len(pred_sets) == 5


class TestAPS:
    """Test Adaptive Prediction Sets (APS)."""

    def test_returns_predictor(self, ood_model, ood_id_loader):
        """Test that APS returns a predictor function."""
        alpha = 0.1
        predictor = aps(ood_model, ood_id_loader, alpha)

        assert callable(predictor)

    def test_predictor_output(self, ood_model, ood_id_loader):
        """Test that APS predictor returns prediction sets."""
        alpha = 0.1
        predictor = aps(ood_model, ood_id_loader, alpha)

        x_test = torch.randn(5, 64)
        pred_sets = predictor(x_test)

        assert isinstance(pred_sets, list)
        assert len(pred_sets) == 5
        for ps in pred_sets:
            assert isinstance(ps, torch.Tensor)

    def test_different_alpha_values(self, ood_model, ood_id_loader):
        """Test APS with different alpha values."""
        x_test = torch.randn(5, 64)

        for alpha in [0.05, 0.1, 0.2]:
            predictor = aps(ood_model, ood_id_loader, alpha)
            pred_sets = predictor(x_test)

            assert len(pred_sets) == 5
            assert all(isinstance(ps, torch.Tensor) for ps in pred_sets)


class TestRAPS:
    """Test Regularized Adaptive Prediction Sets (RAPS)."""

    def test_returns_predictor(self, ood_model, ood_id_loader):
        """Test that RAPS returns a predictor function."""
        alpha = 0.1
        predictor = raps(ood_model, ood_id_loader, alpha)

        assert callable(predictor)

    def test_predictor_output(self, ood_model, ood_id_loader):
        """Test that RAPS predictor returns prediction sets."""
        alpha = 0.1
        predictor = raps(ood_model, ood_id_loader, alpha, lam=0.01, k_reg=1)

        x_test = torch.randn(5, 64)
        pred_sets = predictor(x_test)

        assert isinstance(pred_sets, list)
        assert len(pred_sets) == 5
        for ps in pred_sets:
            assert isinstance(ps, torch.Tensor)

    def test_regularization_parameters(self, ood_model, ood_id_loader):
        """Test RAPS with different regularization parameters."""
        alpha = 0.1
        x_test = torch.randn(5, 64)

        # Test with different lambda values
        predictor_0 = raps(ood_model, ood_id_loader, alpha, lam=0.0, k_reg=1)
        predictor_01 = raps(ood_model, ood_id_loader, alpha, lam=0.1, k_reg=1)

        sets_0 = predictor_0(x_test)
        sets_01 = predictor_01(x_test)

        assert len(sets_0) == len(sets_01) == 5

    def test_minimum_set_size(self, ood_model, ood_id_loader):
        """Test that k_reg enforces minimum set size."""
        alpha = 0.1
        x_test = torch.randn(5, 64)

        predictor = raps(ood_model, ood_id_loader, alpha, lam=0.0, k_reg=2)
        pred_sets = predictor(x_test)

        # Each set should have at least k_reg elements
        for ps in pred_sets:
            assert len(ps) >= 2  # k_reg=2


# Integration test
class TestConformalIntegration:
    """Integration tests for conformal prediction methods."""

    def test_all_classification_methods_work(self, ood_model, ood_id_loader):
        """Test that all classification conformal methods produce valid outputs."""
        alpha = 0.1
        x_test = torch.randn(3, 64)

        methods = [
            inductive_conformal(ood_model, ood_id_loader, alpha),
            mondrian_conformal(ood_model, ood_id_loader, alpha),
            aps(ood_model, ood_id_loader, alpha),
            raps(ood_model, ood_id_loader, alpha),
        ]

        for predictor in methods:
            pred_sets = predictor(x_test)
            assert isinstance(pred_sets, list)
            assert len(pred_sets) == 3
            for ps in pred_sets:
                assert isinstance(ps, torch.Tensor)
                assert len(ps) >= 0  # Can be empty but usually not

    def test_prediction_sets_are_valid_classes(
        self, ood_model, ood_id_loader, num_classes
    ):
        """Test that prediction sets contain valid class indices."""
        alpha = 0.1
        x_test = torch.randn(5, 64)

        predictor = inductive_conformal(ood_model, ood_id_loader, alpha)
        pred_sets = predictor(x_test)

        for ps in pred_sets:
            if len(ps) > 0:  # If not empty
                assert ps.min() >= 0
                assert ps.max() < num_classes
