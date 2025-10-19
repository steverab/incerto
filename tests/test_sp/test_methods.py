"""
Tests for selective prediction methods.

API:
- SelfAdaptiveTraining(backbone, num_classes, alpha_start=0.0, alpha_end=0.9, warmup_epochs=5)
- .sat_loss(logits, targets, alpha) -> loss
- .get_alpha(epoch, total_epochs) -> alpha value
"""

import pytest
import torch
import torch.nn as nn

from incerto.sp import SelfAdaptiveTraining


class TestSelfAdaptiveTraining:
    """Test Self-Adaptive Training (SAT)."""

    def test_initialization(self, simple_model, num_classes):
        """Test SAT can be initialized."""
        sat = SelfAdaptiveTraining(simple_model, num_classes)
        assert sat is not None
        assert sat.num_classes == num_classes
        assert sat.backbone is simple_model

    def test_forward(self, simple_model, num_classes):
        """Test forward pass through model."""
        sat = SelfAdaptiveTraining(simple_model, num_classes)
        x = torch.randn(16, 2)  # simple_model expects 2D input

        logits = sat(x)

        assert logits.shape == (16, num_classes)
        assert torch.isfinite(logits).all()

    def test_sat_loss(
        self, simple_model, multiclass_logits, multiclass_labels, num_classes
    ):
        """Test SAT loss computation."""
        sat = SelfAdaptiveTraining(simple_model, num_classes)

        # Test with different alpha values
        for alpha in [0.0, 0.3, 0.7, 0.9]:
            loss = sat.sat_loss(multiclass_logits, multiclass_labels, alpha)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert loss.item() >= 0

    def test_get_alpha(self, simple_model, num_classes):
        """Test alpha scheduling."""
        sat = SelfAdaptiveTraining(
            simple_model, num_classes, alpha_start=0.0, alpha_end=0.9, warmup_epochs=5
        )

        # Early epochs (warmup)
        alpha_early = sat.get_alpha(epoch=2, total_epochs=100)
        assert alpha_early == 0.0  # Still in warmup

        # After warmup
        alpha_mid = sat.get_alpha(epoch=50, total_epochs=100)
        assert 0.0 < alpha_mid < 0.9  # Should be between start and end

        # Late epochs
        alpha_late = sat.get_alpha(epoch=99, total_epochs=100)
        assert abs(alpha_late - 0.9) < 0.1  # Should be close to alpha_end

    def test_alpha_zero_equals_ce(
        self, simple_model, multiclass_logits, multiclass_labels, num_classes
    ):
        """Test that alpha=0 equals standard cross-entropy."""
        sat = SelfAdaptiveTraining(simple_model, num_classes)

        loss_sat = sat.sat_loss(multiclass_logits, multiclass_labels, alpha=0.0)
        loss_ce = nn.CrossEntropyLoss()(multiclass_logits, multiclass_labels)

        assert torch.allclose(loss_sat, loss_ce, atol=1e-5)

    def test_gradient_flow(
        self, simple_model, multiclass_logits, multiclass_labels, num_classes
    ):
        """Test gradients flow through SAT loss."""
        sat = SelfAdaptiveTraining(simple_model, num_classes)
        logits = multiclass_logits.clone().requires_grad_(True)

        loss = sat.sat_loss(logits, multiclass_labels, alpha=0.5)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_full_training_workflow(self, simple_model, num_classes):
        """Test a complete training workflow with SAT."""
        sat = SelfAdaptiveTraining(simple_model, num_classes, warmup_epochs=2)
        optimizer = torch.optim.SGD(sat.parameters(), lr=0.01)

        # Simulate a few training steps
        for epoch in range(3):
            # Get current alpha
            alpha = sat.get_alpha(epoch=epoch, total_epochs=10)

            # Create batch
            x = torch.randn(16, 2)
            y = torch.randint(0, num_classes, (16,))

            # Forward pass
            optimizer.zero_grad()
            logits = sat(x)

            # Compute SAT loss
            loss = sat.sat_loss(logits, y, alpha)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Verify loss is valid
            assert loss.item() >= 0
            assert torch.isfinite(loss)
