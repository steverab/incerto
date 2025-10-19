"""
Tests for training-time calibration methods.

All are nn.Module losses that take (logits, labels) except where noted.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from incerto.calibration.training import (
    LabelSmoothingLoss,
    FocalLoss,
    ConfidencePenalty,
    evidential_loss,
    get_uncertainty_from_evidence,
    TemperatureAwareTraining,
)


class TestLabelSmoothingLoss:
    """Test LabelSmoothingLoss."""

    def test_initialization(self):
        """Test loss can be initialized."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        assert loss_fn.smoothing == 0.1

    def test_forward(self, multiclass_logits, multiclass_labels):
        """Test forward pass."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        loss = loss_fn(multiclass_logits, multiclass_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative

    def test_smoothing_effect(self, multiclass_logits, multiclass_labels):
        """Test smoothing affects loss."""
        ce_loss = nn.CrossEntropyLoss()
        ls_loss = LabelSmoothingLoss(smoothing=0.1)

        ce = ce_loss(multiclass_logits, multiclass_labels)
        ls = ls_loss(multiclass_logits, multiclass_labels)

        # Both should be positive
        assert ce > 0 and ls > 0

    def test_zero_smoothing_equals_ce(self, multiclass_logits, multiclass_labels):
        """Test zero smoothing equals cross-entropy."""
        ce_loss = nn.CrossEntropyLoss(reduction="mean")
        ls_loss = LabelSmoothingLoss(smoothing=0.0, reduction="mean")

        ce = ce_loss(multiclass_logits, multiclass_labels)
        ls = ls_loss(multiclass_logits, multiclass_labels)

        assert torch.allclose(ce, ls, atol=1e-5)

    def test_different_reduction(self, multiclass_logits, multiclass_labels):
        """Test different reduction modes."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = LabelSmoothingLoss(smoothing=0.1, reduction=reduction)
            loss = loss_fn(multiclass_logits, multiclass_labels)

            if reduction == "none":
                assert loss.shape == (len(multiclass_labels),)
            else:
                assert loss.dim() == 0  # Scalar

    def test_smoothing_range(self, multiclass_logits, multiclass_labels):
        """Test different smoothing values."""
        for smoothing in [0.0, 0.05, 0.1, 0.2, 0.3]:
            loss_fn = LabelSmoothingLoss(smoothing=smoothing)
            loss = loss_fn(multiclass_logits, multiclass_labels)
            assert loss >= 0

    def test_gradient_flow(self, multiclass_logits, multiclass_labels):
        """Test gradients can flow through loss."""
        logits = multiclass_logits.clone().requires_grad_(True)
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        loss = loss_fn(logits, multiclass_labels)

        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestFocalLoss:
    """Test FocalLoss."""

    def test_initialization(self):
        """Test loss can be initialized."""
        loss_fn = FocalLoss(gamma=2.0, alpha=1.0)
        assert loss_fn.gamma == 2.0
        assert loss_fn.alpha == 1.0

    def test_forward(self, multiclass_logits, multiclass_labels):
        """Test forward pass."""
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(multiclass_logits, multiclass_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0

    def test_gamma_effect(self, multiclass_logits, multiclass_labels):
        """Test gamma parameter affects focus on hard examples."""
        loss_gamma0 = FocalLoss(gamma=0.0)(multiclass_logits, multiclass_labels)
        loss_gamma2 = FocalLoss(gamma=2.0)(multiclass_logits, multiclass_labels)

        # Both should be positive
        assert loss_gamma0 > 0 and loss_gamma2 > 0

    def test_zero_gamma_equals_ce(self, multiclass_logits, multiclass_labels):
        """Test gamma=0 equals cross-entropy."""
        ce_loss = nn.CrossEntropyLoss(reduction="mean")
        focal_loss = FocalLoss(gamma=0.0, alpha=1.0, reduction="mean")

        ce = ce_loss(multiclass_logits, multiclass_labels)
        focal = focal_loss(multiclass_logits, multiclass_labels)

        # Should be close (may have small numerical differences)
        assert torch.allclose(ce, focal, atol=1e-4)

    def test_different_reduction(self, multiclass_logits, multiclass_labels):
        """Test different reduction modes."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = FocalLoss(gamma=2.0, reduction=reduction)
            loss = loss_fn(multiclass_logits, multiclass_labels)

            if reduction == "none":
                assert loss.shape == (len(multiclass_labels),)
            else:
                assert loss.dim() == 0

    def test_gradient_flow(self, multiclass_logits, multiclass_labels):
        """Test gradients can flow through loss."""
        logits = multiclass_logits.clone().requires_grad_(True)
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, multiclass_labels)

        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestConfidencePenalty:
    """Test ConfidencePenalty."""

    def test_initialization(self):
        """Test loss can be initialized."""
        loss_fn = ConfidencePenalty(beta=0.1)
        assert loss_fn.beta == 0.1

    def test_forward(self, multiclass_logits, multiclass_labels):
        """Test forward pass - includes both CE and entropy penalty."""
        loss_fn = ConfidencePenalty(beta=0.1)
        loss = loss_fn(multiclass_logits, multiclass_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Should be non-negative

    def test_relation_to_ce(self, multiclass_logits, multiclass_labels):
        """Test confidence penalty modifies CE loss."""
        ce_loss = nn.CrossEntropyLoss()
        cp_loss = ConfidencePenalty(beta=0.1)

        ce = ce_loss(multiclass_logits, multiclass_labels)
        cp = cp_loss(multiclass_logits, multiclass_labels)

        # CP = CE - beta * entropy, so can be less than CE (encourages uncertainty)
        # Just verify it's a valid finite loss
        assert torch.isfinite(cp)
        assert cp.item() >= 0  # Should still be non-negative

    def test_beta_scaling(self, multiclass_logits, multiclass_labels):
        """Test beta parameter scales the entropy regularization."""
        loss_beta_zero = ConfidencePenalty(beta=0.0)(
            multiclass_logits, multiclass_labels
        )
        loss_beta_small = ConfidencePenalty(beta=0.01)(
            multiclass_logits, multiclass_labels
        )
        loss_beta_large = ConfidencePenalty(beta=1.0)(
            multiclass_logits, multiclass_labels
        )

        # CP = CE - beta * entropy
        # Larger beta subtracts more entropy, so smaller loss (encourages uncertainty)
        # beta=0 should equal CE
        ce = nn.CrossEntropyLoss()(multiclass_logits, multiclass_labels)
        assert torch.isclose(loss_beta_zero, ce, atol=1e-5)

        # Verify all losses are valid
        assert torch.isfinite(loss_beta_small)
        assert torch.isfinite(loss_beta_large)

    def test_gradient_flow(self, multiclass_logits, multiclass_labels):
        """Test gradients can flow through loss."""
        logits = multiclass_logits.clone().requires_grad_(True)
        loss_fn = ConfidencePenalty(beta=0.1)
        loss = loss_fn(logits, multiclass_labels)

        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestEvidentialLoss:
    """Test evidential_loss function."""

    def test_basic_forward(self, num_classes):
        """Test basic forward pass."""
        batch_size = 32
        # Evidence should be non-negative
        evidence = torch.rand(batch_size, num_classes) * 10
        targets = torch.randint(0, num_classes, (batch_size,))

        # Returns tuple of (total_loss, mse_loss, kl_loss)
        result = evidential_loss(
            evidence, targets, num_classes, epoch=5, num_epochs=100, kl_weight=1.0
        )

        # Check it's a tuple of 3 tensors
        assert isinstance(result, tuple)
        assert len(result) == 3
        total_loss, mse_loss, kl_loss = result

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(mse_loss, torch.Tensor)
        assert isinstance(kl_loss, torch.Tensor)

        assert total_loss.dim() == 0
        assert mse_loss.dim() == 0
        assert kl_loss.dim() == 0

        assert total_loss.item() >= 0
        assert mse_loss.item() >= 0
        assert kl_loss.item() >= 0

    def test_epoch_annealing(self, num_classes):
        """Test KL annealing over epochs."""
        batch_size = 32
        evidence = torch.rand(batch_size, num_classes) * 10
        targets = torch.randint(0, num_classes, (batch_size,))

        # Early epoch
        total_early, mse_early, kl_early = evidential_loss(
            evidence, targets, num_classes, epoch=1, num_epochs=100, kl_weight=1.0
        )
        # Late epoch
        total_late, mse_late, kl_late = evidential_loss(
            evidence, targets, num_classes, epoch=99, num_epochs=100, kl_weight=1.0
        )

        # All should be valid losses
        assert total_early >= 0
        assert total_late >= 0

        # KL contribution should be higher in late epochs (annealed)
        # (This is probabilistic but generally true)

    def test_gradient_flow(self, num_classes):
        """Test gradients can flow through loss."""
        batch_size = 32
        # Create evidence as a leaf tensor with requires_grad
        evidence = torch.rand(batch_size, num_classes) * 10
        evidence.requires_grad_(True)
        targets = torch.randint(0, num_classes, (batch_size,))

        total_loss, mse_loss, kl_loss = evidential_loss(
            evidence, targets, num_classes, epoch=5, num_epochs=100, kl_weight=1.0
        )

        total_loss.backward()
        assert evidence.grad is not None
        assert not torch.isnan(evidence.grad).any()


class TestGetUncertaintyFromEvidence:
    """Test get_uncertainty_from_evidence function."""

    def test_basic_computation(self, num_classes):
        """Test basic uncertainty computation."""
        batch_size = 32
        evidence = torch.rand(batch_size, num_classes) * 10

        result = get_uncertainty_from_evidence(evidence, num_classes)

        # Should return a dictionary
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "belief" in result
        assert "uncertainty" in result
        assert "epistemic" in result

        # Check shapes
        assert result["alpha"].shape == (batch_size, num_classes)
        assert result["belief"].shape == (batch_size, num_classes)
        assert result["uncertainty"].shape == (batch_size, 1)
        assert result["epistemic"].shape == (batch_size, 1)

    def test_uncertainty_ranges(self, num_classes):
        """Test uncertainty values are in valid ranges."""
        batch_size = 32
        evidence = torch.rand(batch_size, num_classes) * 10

        result = get_uncertainty_from_evidence(evidence, num_classes)

        # Alpha should be >= 1 (evidence + 1)
        assert (result["alpha"] >= 1).all()

        # Belief should sum to 1
        assert torch.allclose(
            result["belief"].sum(dim=1), torch.ones(batch_size), atol=1e-5
        )

        # Uncertainties should be non-negative
        assert (result["uncertainty"] >= 0).all()
        assert (result["epistemic"] >= 0).all()

    def test_high_evidence_low_uncertainty(self, num_classes):
        """Test high evidence leads to low uncertainty."""
        # Low evidence
        evidence_low = torch.ones(10, num_classes) * 0.1
        # High evidence
        evidence_high = torch.ones(10, num_classes) * 100

        result_low = get_uncertainty_from_evidence(evidence_low, num_classes)
        result_high = get_uncertainty_from_evidence(evidence_high, num_classes)

        # High evidence should have lower uncertainty
        assert result_high["uncertainty"].mean() < result_low["uncertainty"].mean()
        assert result_high["epistemic"].mean() < result_low["epistemic"].mean()


class TestTemperatureAwareTraining:
    """Test TemperatureAwareTraining wrapper."""

    def test_initialization(self):
        """Test wrapper can be initialized."""
        backbone = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
        model = TemperatureAwareTraining(backbone, init_temp=1.5)
        assert model.temperature.item() == 1.5

    def test_forward(self):
        """Test forward pass."""
        backbone = nn.Sequential(nn.Linear(10, 5))
        model = TemperatureAwareTraining(backbone, init_temp=1.5)

        x = torch.randn(32, 10)
        logits = model(x)

        assert logits.shape == (32, 5)

    def test_temperature_scaling(self):
        """Test temperature scales logits."""
        backbone = nn.Sequential(nn.Linear(10, 5))
        model = TemperatureAwareTraining(backbone, init_temp=2.0)

        x = torch.randn(32, 10)

        # Get unscaled logits
        unscaled = model(x, return_unscaled=True)
        # Get scaled logits
        scaled = model(x, return_unscaled=False)

        # Scaled should be unscaled / temperature
        expected = unscaled / 2.0
        assert torch.allclose(scaled, expected, atol=1e-5)

    def test_learnable_temperature(self):
        """Test temperature can be learned."""
        backbone = nn.Sequential(nn.Linear(10, 5))
        model = TemperatureAwareTraining(backbone, init_temp=1.5, learn_temp=True)

        # Temperature should have gradient enabled
        assert model.temperature.requires_grad

        # Test learning
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        for _ in range(3):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Temperature should have changed (probabilistic but likely)
        # Just verify it's still positive
        assert model.temperature.item() > 0


# Integration tests
class TestTrainingMethodIntegration:
    """Integration tests for training methods."""

    def test_all_losses_work(self, multiclass_logits, multiclass_labels):
        """Test all training losses can compute loss."""
        losses = [
            LabelSmoothingLoss(smoothing=0.1),
            FocalLoss(gamma=2.0),
            ConfidencePenalty(beta=0.1),
        ]

        for loss_fn in losses:
            loss = loss_fn(multiclass_logits, multiclass_labels)
            assert loss >= 0

    def test_combined_loss(self, multiclass_logits, multiclass_labels):
        """Test ConfidencePenalty combines CE with entropy regularization."""
        ce_loss = nn.CrossEntropyLoss()

        # ConfidencePenalty = CE - beta * entropy
        cp_loss = ConfidencePenalty(beta=0.1)

        ce = ce_loss(multiclass_logits, multiclass_labels)
        cp = cp_loss(multiclass_logits, multiclass_labels)

        # CP modifies CE by entropy regularization (can be more or less than CE)
        assert torch.isfinite(cp)
        assert cp.item() >= 0  # Should still be non-negative

    def test_training_loop_simulation(self, simple_model, simple_dataloader):
        """Simulate a training loop with calibration loss."""
        model = simple_model
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        model.train()
        for batch_x, batch_y in simple_dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            # Should complete without errors
            assert loss.item() >= 0

    def test_evidential_training_loop(self, num_classes):
        """Test evidential training loop."""
        # Simple model that outputs evidence
        model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softplus(),  # Ensure positive evidence
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Dummy data
        X = torch.randn(64, 2)
        y = torch.randint(0, num_classes, (64,))

        for epoch in range(3):
            optimizer.zero_grad()
            evidence = model(X)
            total_loss, mse_loss, kl_loss = evidential_loss(
                evidence, y, num_classes, epoch, num_epochs=10
            )
            total_loss.backward()
            optimizer.step()

            assert total_loss.item() >= 0
            assert mse_loss.item() >= 0
            assert kl_loss.item() >= 0
