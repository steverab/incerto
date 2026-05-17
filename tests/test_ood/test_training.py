"""
Tests for OOD training methods.

Correct API usage:
- mixup_data(x, y, alpha=1.0) -> (mixed_x, y_a, y_b, lambda)
- mixup_criterion(criterion, pred, y_a, y_b, lam) -> loss
- OutlierExposureLoss(lambda_oe=0.5).forward(logits_in, targets_in, logits_out=None)
- EnergyRegularizedLoss(lambda_energy=0.1, margin=10.0).forward(logits_in, targets_in, logits_out=None)
- CutMix(alpha=1.0)(x, y) -> (mixed_x, y_a, y_b, lambda)
"""

import numpy as np
import torch
import torch.nn as nn

from incerto.ood.training import (
    CutMix,
    EnergyRegularizedLoss,
    OutlierExposureLoss,
    mixup_criterion,
    mixup_data,
)


class TestMixup:
    """Test Mixup data augmentation."""

    def test_mixup_data_shape(self, set_seed):
        """Test mixup preserves data shape."""
        x = torch.randn(32, 3, 28, 28)
        y = torch.randint(0, 10, (32,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, float)
        assert 0 <= lam <= 1

    def test_mixup_lambda_range(self, set_seed):
        """Test lambda is in [0, 1]."""
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            _, _, _, lam = mixup_data(x, y, alpha=alpha)
            assert 0 <= lam <= 1

    def test_mixup_deterministic_with_seed(self):
        """Test mixup is deterministic with same seed."""
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        np.random.seed(42)
        torch.manual_seed(42)
        mixed_x1, y_a1, y_b1, lam1 = mixup_data(x, y, alpha=1.0)

        np.random.seed(42)
        torch.manual_seed(42)
        mixed_x2, y_a2, y_b2, lam2 = mixup_data(x, y, alpha=1.0)

        assert torch.allclose(mixed_x1, mixed_x2)
        assert torch.equal(y_a1, y_a2)
        assert torch.equal(y_b1, y_b2)
        assert lam1 == lam2

    def test_mixup_criterion(self, set_seed):
        """Test mixup criterion."""
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(32, 10)
        y_a = torch.randint(0, 10, (32,))
        y_b = torch.randint(0, 10, (32,))
        lam = 0.5

        loss = mixup_criterion(criterion, pred, y_a, y_b, lam)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_mixup_criterion_lambda_0(self, set_seed):
        """Test mixup criterion with lambda=0 equals loss on y_b."""
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(32, 10)
        y_a = torch.randint(0, 10, (32,))
        y_b = torch.randint(0, 10, (32,))

        loss_mixup = mixup_criterion(criterion, pred, y_a, y_b, lam=0.0)
        loss_b = criterion(pred, y_b)

        assert torch.allclose(loss_mixup, loss_b, atol=1e-5)

    def test_mixup_criterion_lambda_1(self, set_seed):
        """Test mixup criterion with lambda=1 equals loss on y_a."""
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(32, 10)
        y_a = torch.randint(0, 10, (32,))
        y_b = torch.randint(0, 10, (32,))

        loss_mixup = mixup_criterion(criterion, pred, y_a, y_b, lam=1.0)
        loss_a = criterion(pred, y_a)

        assert torch.allclose(loss_mixup, loss_a, atol=1e-5)


class TestOutlierExposureLoss:
    """Test Outlier Exposure loss."""

    def test_initialization(self):
        """Test loss can be initialized."""
        loss_fn = OutlierExposureLoss(lambda_oe=0.5)
        assert loss_fn.lambda_oe == 0.5

    def test_forward_id_only(self, multiclass_logits, multiclass_labels):
        """Test forward pass with ID data only."""
        loss_fn = OutlierExposureLoss(lambda_oe=0.5)
        loss = loss_fn(multiclass_logits, multiclass_labels, logits_out=None)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_with_ood(self, multiclass_logits, multiclass_labels, num_classes):
        """Test forward pass with both ID and OOD data."""
        loss_fn = OutlierExposureLoss(lambda_oe=0.5)
        logits_out = torch.randn(20, num_classes)  # OOD data

        loss = loss_fn(multiclass_logits, multiclass_labels, logits_out)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_lambda_zero_equals_ce(self, multiclass_logits, multiclass_labels):
        """Test lambda_oe=0 equals CE loss."""
        loss_fn = OutlierExposureLoss(lambda_oe=0.0)
        loss_oe = loss_fn(multiclass_logits, multiclass_labels, logits_out=None)

        ce_loss = nn.CrossEntropyLoss()(multiclass_logits, multiclass_labels)

        assert torch.allclose(loss_oe, ce_loss, atol=1e-5)

    def test_gradient_flow(self, multiclass_logits, multiclass_labels):
        """Test gradients can flow through loss."""
        logits = multiclass_logits.clone().requires_grad_(True)
        loss_fn = OutlierExposureLoss()
        loss = loss_fn(logits, multiclass_labels)

        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestEnergyRegularizedLoss:
    """Test Energy Regularized loss."""

    def test_initialization(self):
        """Test loss can be initialized."""
        loss_fn = EnergyRegularizedLoss(lambda_energy=0.1, margin=10.0)
        assert loss_fn.lambda_energy == 0.1
        assert loss_fn.margin == 10.0

    def test_forward_id_only(self, multiclass_logits, multiclass_labels):
        """Test forward pass with ID data only."""
        loss_fn = EnergyRegularizedLoss(lambda_energy=0.1)
        loss = loss_fn(multiclass_logits, multiclass_labels, logits_out=None)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_with_ood(self, multiclass_logits, multiclass_labels, num_classes):
        """Test forward pass with both ID and OOD data."""
        loss_fn = EnergyRegularizedLoss(lambda_energy=0.1, margin=10.0)
        # OOD data - same batch size as ID data
        logits_out = torch.randn(len(multiclass_logits), num_classes)

        loss = loss_fn(multiclass_logits, multiclass_labels, logits_out)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_lambda_zero_equals_ce(self, multiclass_logits, multiclass_labels):
        """Test lambda_energy=0 equals CE loss."""
        loss_fn = EnergyRegularizedLoss(lambda_energy=0.0)
        loss_energy = loss_fn(multiclass_logits, multiclass_labels, logits_out=None)

        ce_loss = nn.CrossEntropyLoss()(multiclass_logits, multiclass_labels)

        assert torch.allclose(loss_energy, ce_loss, atol=1e-5)

    def test_energy_regularization_effect(self, multiclass_logits, multiclass_labels, num_classes):
        """Test energy regularization affects loss."""
        # OOD data - same batch size as ID data
        logits_out = torch.randn(len(multiclass_logits), num_classes)

        loss_no_reg = EnergyRegularizedLoss(lambda_energy=0.0)
        loss_with_reg = EnergyRegularizedLoss(lambda_energy=0.5)

        loss_0 = loss_no_reg(multiclass_logits, multiclass_labels, logits_out)
        loss_1 = loss_with_reg(multiclass_logits, multiclass_labels, logits_out)

        # Should be different
        assert not torch.allclose(loss_0, loss_1, atol=1e-5)

    def test_gradient_flow(self, multiclass_logits, multiclass_labels):
        """Test gradients can flow through loss."""
        logits = multiclass_logits.clone().requires_grad_(True)
        loss_fn = EnergyRegularizedLoss(lambda_energy=0.1)

        loss = loss_fn(logits, multiclass_labels)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestCutMix:
    """Test CutMix augmentation."""

    def test_initialization(self):
        """Test CutMix can be initialized."""
        cutmix = CutMix(alpha=1.0)
        assert cutmix.alpha == 1.0

    def test_forward_shape(self, set_seed):
        """Test CutMix preserves shape."""
        cutmix = CutMix(alpha=1.0)
        x = torch.randn(32, 3, 28, 28)
        y = torch.randint(0, 10, (32,))

        mixed_x, y_a, y_b, lam = cutmix(x, y)

        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, float)
        assert 0 <= lam <= 1

    def test_different_alpha(self, set_seed):
        """Test CutMix with different alpha values."""
        x = torch.randn(32, 3, 28, 28)
        y = torch.randint(0, 10, (32,))

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            cutmix = CutMix(alpha=alpha)
            mixed_x, y_a, y_b, lam = cutmix(x, y)

            assert mixed_x.shape == x.shape
            assert 0 <= lam <= 1

    def test_deterministic_with_seed(self):
        """Test CutMix is deterministic with same seed."""
        x = torch.randn(32, 3, 28, 28)
        y = torch.randint(0, 10, (32,))

        cutmix = CutMix(alpha=1.0)

        np.random.seed(42)
        torch.manual_seed(42)
        mixed_x1, y_a1, y_b1, lam1 = cutmix(x, y)

        np.random.seed(42)
        torch.manual_seed(42)
        mixed_x2, y_a2, y_b2, lam2 = cutmix(x, y)

        assert torch.allclose(mixed_x1, mixed_x2)
        assert torch.equal(y_a1, y_a2)
        assert torch.equal(y_b1, y_b2)
        assert lam1 == lam2


# Integration tests
class TestOODTrainingIntegration:
    """Integration tests for OOD training methods."""

    def test_mixup_training_loop(self, simple_model, simple_dataloader):
        """Simulate training loop with mixup."""
        model = simple_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # One training step with mixup
        for x, y in simple_dataloader:
            optimizer.zero_grad()

            # Apply mixup
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

            # Forward pass
            outputs = model(mixed_x)

            # Mixup loss
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Just verify one iteration works
            assert loss.item() >= 0
            break

    def test_cutmix_training_loop(self):
        """Simulate training loop with CutMix."""
        # Simple 2D Conv model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        cutmix = CutMix(alpha=1.0)

        # Create simple batch
        x = torch.randn(8, 3, 28, 28)
        y = torch.randint(0, 10, (8,))

        optimizer.zero_grad()

        # Apply CutMix
        mixed_x, y_a, y_b, lam = cutmix(x, y)

        # Forward pass
        outputs = model(mixed_x)

        # CutMix loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_outlier_exposure_training(self, simple_model, simple_dataloader):
        """Test training with outlier exposure."""
        model = simple_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = OutlierExposureLoss(lambda_oe=0.5)

        for x, y in simple_dataloader:
            optimizer.zero_grad()

            # Get ID logits
            logits_in = model(x)

            # Create synthetic OOD data (just noise for testing)
            x_out = torch.randn_like(x) * 2.0
            logits_out = model(x_out)

            # OE loss
            loss = criterion(logits_in, y, logits_out)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Just verify one iteration works
            assert loss.item() >= 0
            break
