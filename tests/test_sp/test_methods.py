"""
Tests for selective prediction methods.

API:
- SoftmaxThreshold(backbone) — confidence via MSP
- DeepGambler(backbone, num_classes) — extra abstain logit
- SelectiveNet(backbone, num_classes) — dedicated selection head g(x)
- SelfAdaptiveTraining(backbone, num_classes, alpha_start=0.0, alpha_end=0.9, warmup_epochs=5)
"""

import pytest
import torch
import torch.nn as nn

from incerto.sp import (
    DeepGambler,
    SelectiveNet,
    SelfAdaptiveTraining,
    SoftmaxThreshold,
    accuracy_coverage_curve,
    aurc,
    coverage,
    make,
    plot_risk_coverage,
    risk,
)
from incerto.sp.methods import _infer_output_dim


class TestSoftmaxThreshold:
    """Test SoftmaxThreshold selective predictor."""

    def test_forward_logits_only(self, simple_model):
        """Test forward pass returns logits."""
        selector = SoftmaxThreshold(simple_model)
        x = torch.randn(16, 2)
        logits = selector(x)

        assert logits.shape == (16, 10)  # default num_classes from simple_model
        assert torch.isfinite(logits).all()

    def test_forward_with_confidence(self, simple_model):
        """Test forward pass with return_confidence=True."""
        selector = SoftmaxThreshold(simple_model)
        x = torch.randn(16, 2)
        logits, conf = selector(x, return_confidence=True)

        assert logits.shape[0] == 16
        assert conf.shape == (16,)
        # Confidence should be in [0, 1] (max softmax prob)
        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_confidence_is_max_softmax(self, simple_model):
        """Test that confidence equals max softmax probability."""
        selector = SoftmaxThreshold(simple_model)
        x = torch.randn(8, 2)
        logits, conf = selector(x, return_confidence=True)

        expected = torch.softmax(logits, dim=-1).max(dim=-1).values
        assert torch.allclose(conf, expected)

    def test_reject(self, simple_model):
        """Test reject utility returns correct boolean mask."""
        selector = SoftmaxThreshold(simple_model)
        x = torch.randn(16, 2)
        _, conf = selector(x, return_confidence=True)

        threshold = 0.5
        rejected = selector.reject(conf, threshold)

        assert rejected.dtype == torch.bool
        assert rejected.shape == (16,)
        assert (rejected == (conf < threshold)).all()


class TestDeepGambler:
    """Test DeepGambler selective predictor."""

    def test_forward_logits(self):
        """Test forward pass returns correct shape."""
        backbone = nn.Linear(10, 32)
        gambler = DeepGambler(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        logits = gambler(x)

        # DeepGambler head outputs num_classes + 1 logits but
        # _forward_logits returns all of them
        assert logits.shape == (16, 6)  # 5 classes + 1 abstain

    def test_forward_with_confidence(self):
        """Test confidence is 1 - P(abstain)."""
        backbone = nn.Linear(10, 32)
        gambler = DeepGambler(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        logits, conf = gambler(x, return_confidence=True)

        assert conf.shape == (16,)
        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_confidence_from_logits(self):
        """Test confidence_from_logits computes 1 - P(abstain)."""
        backbone = nn.Linear(10, 32)
        gambler = DeepGambler(backbone, num_classes=5, num_features=32)
        x = torch.randn(8, 10)
        logits = gambler(x)  # (8, 6) — 5 classes + 1 abstain

        conf = gambler.confidence_from_logits(logits)
        probs = torch.softmax(logits, dim=-1)
        expected = 1.0 - probs[:, -1]
        assert torch.allclose(conf, expected)

    def test_gambler_loss_basic(self):
        """Test gambler's loss produces finite positive scalar."""
        backbone = nn.Linear(10, 32)
        gambler = DeepGambler(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))
        logits = gambler(x)

        loss = gambler.gambler_loss(logits, y, reward=2.2)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gambler_loss_gradient_flow(self):
        """Test gradients flow through gambler's loss."""
        backbone = nn.Linear(10, 32)
        gambler = DeepGambler(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))

        logits = gambler(x)
        loss = gambler.gambler_loss(logits, y)
        loss.backward()

        # Check gradients exist on head parameters
        assert gambler.head.weight.grad is not None
        assert not torch.isnan(gambler.head.weight.grad).any()

    def test_gambler_loss_higher_reward_less_abstention(self):
        """Higher reward should increase confidence (less abstention)."""
        backbone = nn.Linear(10, 32)
        gambler = DeepGambler(backbone, num_classes=5, num_features=32)
        optimizer = torch.optim.Adam(gambler.parameters(), lr=0.01)
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))

        # Train a few steps with high reward (penalises abstention)
        for _ in range(20):
            optimizer.zero_grad()
            logits = gambler(x)
            loss = gambler.gambler_loss(logits, y, reward=6.0)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, conf = gambler(x, return_confidence=True)
        # With high reward, model should lean away from abstaining
        assert conf.mean() > 0.5


class TestSelectiveNet:
    """Test SelectiveNet selective predictor."""

    def test_forward_logits(self):
        """Test forward pass returns class logits (not selection prob)."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        logits = snet(x)

        assert logits.shape == (16, 5)
        assert torch.isfinite(logits).all()

    def test_forward_with_confidence(self):
        """Test confidence is the selection head g(x)."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        logits, conf = snet(x, return_confidence=True)

        assert logits.shape == (16, 5)
        assert conf.shape == (16,)
        # g(x) uses Sigmoid → output in [0, 1]
        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_confidence_from_logits_raises(self):
        """Test that confidence_from_logits raises NotImplementedError."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32)
        logits = torch.randn(8, 5)

        with pytest.raises(NotImplementedError, match="selection head"):
            snet.confidence_from_logits(logits)

    def test_reject_with_selection_head(self):
        """Test reject works with selection head confidence."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        _, conf = snet(x, return_confidence=True)

        rejected = snet.reject(conf, threshold=0.5)
        assert rejected.dtype == torch.bool
        assert rejected.shape == (16,)

    def test_selective_loss_basic(self):
        """Test SelectiveNet loss produces finite scalar."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))

        logits, sel = snet(x, return_confidence=True)
        loss = snet.selective_loss(logits, y, sel, coverage_target=0.8)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_selective_loss_gradient_flow(self):
        """Test gradients flow through selective loss."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32)
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))

        logits, sel = snet(x, return_confidence=True)
        loss = snet.selective_loss(logits, y, sel)
        loss.backward()

        assert snet.h.weight.grad is not None
        assert snet.g[0].weight.grad is not None

    def test_selective_loss_default_coverage(self):
        """Test selective loss uses self.alpha when coverage_target not given."""
        backbone = nn.Linear(10, 32)
        snet = SelectiveNet(backbone, num_classes=5, num_features=32, alpha=0.7)
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))

        logits, sel = snet(x, return_confidence=True)
        # Should not raise — uses self.alpha=0.7 as default
        loss = snet.selective_loss(logits, y, sel)
        assert torch.isfinite(loss)

    def test_selective_loss_lam_penalty(self):
        """Test that lam controls penalty strength (separate from alpha)."""
        backbone = nn.Linear(10, 32)
        torch.manual_seed(0)
        snet_low = SelectiveNet(backbone, num_classes=5, num_features=32, alpha=0.99, lam=0.0)
        torch.manual_seed(0)
        snet_high = SelectiveNet(backbone, num_classes=5, num_features=32, alpha=0.99, lam=100.0)
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))

        logits_l, sel_l = snet_low(x, return_confidence=True)
        logits_h, sel_h = snet_high(x, return_confidence=True)

        loss_low = snet_low.selective_loss(logits_l, y, sel_l)
        loss_high = snet_high.selective_loss(logits_h, y, sel_h)

        # Same model weights → same selective_loss, but lam=100 adds large penalty
        # (coverage is << 0.99 at init), so loss_high > loss_low
        assert loss_high > loss_low


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

    def test_sat_loss(self, simple_model, multiclass_logits, multiclass_labels, num_classes):
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

    def test_gradient_flow(self, simple_model, multiclass_logits, multiclass_labels, num_classes):
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


# ----------------------------------------------------------------------
#  Metrics
# ----------------------------------------------------------------------
class TestCoverage:
    """Tests for the coverage metric."""

    def test_all_accepted(self):
        """coverage = 1.0 when nothing is rejected."""
        reject_mask = torch.zeros(50, dtype=torch.bool)
        assert coverage(reject_mask).item() == pytest.approx(1.0)

    def test_all_rejected(self):
        """coverage = 0.0 when everything is rejected."""
        reject_mask = torch.ones(50, dtype=torch.bool)
        assert coverage(reject_mask).item() == pytest.approx(0.0)

    def test_partial(self):
        """coverage equals fraction of accepted samples."""
        reject_mask = torch.tensor([True, False, False, True, False])
        assert coverage(reject_mask).item() == pytest.approx(0.6)


class TestRisk:
    """Tests for the risk metric."""

    def test_all_correct_accepted(self):
        """Risk = 0 when all accepted predictions are correct."""
        pred = torch.tensor([0, 1, 2, 3])
        y = torch.tensor([0, 1, 2, 3])
        reject = torch.zeros(4, dtype=torch.bool)
        assert risk(pred, y, reject).item() == pytest.approx(0.0, abs=1e-6)

    def test_all_wrong_accepted(self):
        """Risk = 1 when all accepted predictions are wrong."""
        pred = torch.tensor([1, 0, 3, 2])
        y = torch.tensor([0, 1, 2, 3])
        reject = torch.zeros(4, dtype=torch.bool)
        assert risk(pred, y, reject).item() == pytest.approx(1.0, abs=1e-6)

    def test_rejection_reduces_risk(self):
        """Rejecting wrong predictions should lower risk."""
        pred = torch.tensor([0, 1, 9, 9])  # last two wrong
        y = torch.tensor([0, 1, 2, 3])
        reject_none = torch.zeros(4, dtype=torch.bool)
        reject_wrong = torch.tensor([False, False, True, True])

        risk_no_reject = risk(pred, y, reject_none)
        risk_with_reject = risk(pred, y, reject_wrong)

        assert risk_with_reject < risk_no_reject

    def test_all_rejected_returns_one(self):
        """Risk ~1.0 when all samples are rejected (edge case)."""
        pred = torch.tensor([0, 1, 2])
        y = torch.tensor([0, 1, 2])
        reject = torch.ones(3, dtype=torch.bool)
        assert risk(pred, y, reject).item() == pytest.approx(1.0, abs=1e-6)


class TestAURC:
    """Tests for Area Under Risk-Coverage curve."""

    def test_perfect_model(self):
        """AURC = 0 for a perfectly confident and correct model."""
        n = 100
        sorted_conf = torch.linspace(1, 0, n)
        sorted_errors = torch.zeros(n)
        assert aurc(sorted_conf, sorted_errors).item() == pytest.approx(0.0, abs=1e-6)

    def test_worst_model(self):
        """AURC = 1 when all predictions are wrong."""
        n = 100
        sorted_conf = torch.linspace(1, 0, n)
        sorted_errors = torch.ones(n)
        assert aurc(sorted_conf, sorted_errors).item() == pytest.approx(1.0, abs=0.02)

    def test_aurc_non_negative(self):
        """AURC should always be non-negative."""
        torch.manual_seed(42)
        sorted_conf = torch.rand(50).sort(descending=True).values
        sorted_errors = (torch.rand(50) > 0.7).float()
        assert aurc(sorted_conf, sorted_errors).item() >= 0

    def test_better_ranking_lower_aurc(self):
        """Model that puts errors at low-confidence end has lower AURC."""
        n = 20
        # Good ranking: errors only in low-confidence half
        good_conf = torch.linspace(1, 0, n)
        good_errors = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])

        # Bad ranking: errors only in high-confidence half
        bad_conf = torch.linspace(1, 0, n)
        bad_errors = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])

        assert aurc(good_conf, good_errors) < aurc(bad_conf, bad_errors)


class TestAccuracyCoverageCurve:
    """Tests for accuracy_coverage_curve."""

    def test_output_shapes(self, multiclass_logits, multiclass_labels):
        """Coverage and accuracy tensors have same length as input."""
        cov, acc = accuracy_coverage_curve(multiclass_logits, multiclass_labels)
        n = len(multiclass_labels)
        assert cov.shape == (n,)
        assert acc.shape == (n,)

    def test_coverage_monotonic(self, multiclass_logits, multiclass_labels):
        """Coverage should be monotonically increasing."""
        cov, _ = accuracy_coverage_curve(multiclass_logits, multiclass_labels)
        assert (cov[1:] >= cov[:-1]).all()

    def test_full_coverage_last(self, multiclass_logits, multiclass_labels):
        """Last coverage value should be 1.0."""
        cov, _ = accuracy_coverage_curve(multiclass_logits, multiclass_labels)
        assert cov[-1].item() == pytest.approx(1.0)

    def test_custom_confidence(self, multiclass_logits, multiclass_labels):
        """Accepts externally provided confidence scores."""
        conf = torch.rand(len(multiclass_labels))
        cov, acc = accuracy_coverage_curve(multiclass_logits, multiclass_labels, conf)
        assert cov.shape == (len(multiclass_labels),)
        assert (acc >= 0).all() and (acc <= 1).all()

    def test_accuracy_in_range(self, multiclass_logits, multiclass_labels):
        """Accuracy values must be in [0, 1]."""
        _, acc = accuracy_coverage_curve(multiclass_logits, multiclass_labels)
        assert (acc >= 0).all()
        assert (acc <= 1).all()


# ----------------------------------------------------------------------
#  Visualization
# ----------------------------------------------------------------------
class TestPlotRiskCoverage:
    """Tests for plot_risk_coverage visualization."""

    def test_returns_axes(self, multiclass_logits, multiclass_labels):
        """plot_risk_coverage should return a matplotlib Axes object."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_risk_coverage(multiclass_logits, multiclass_labels, ax=ax)
        assert result is ax
        plt.close(fig)

    def test_creates_axes_if_none(self, multiclass_logits, multiclass_labels):
        """Should create axes when not provided."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = plot_risk_coverage(multiclass_logits, multiclass_labels)
        assert ax is not None
        plt.close("all")

    def test_show_aurc_in_title(self, multiclass_logits, multiclass_labels):
        """AURC value should appear in title when show_aurc=True."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plot_risk_coverage(multiclass_logits, multiclass_labels, ax=ax, show_aurc=True)
        assert "AURC" in ax.get_title()
        plt.close(fig)

    def test_no_aurc_in_title(self, multiclass_logits, multiclass_labels):
        """AURC should not appear in title when show_aurc=False."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plot_risk_coverage(multiclass_logits, multiclass_labels, ax=ax, show_aurc=False)
        assert "AURC" not in ax.get_title()
        plt.close(fig)


# ----------------------------------------------------------------------
#  Factory
# ----------------------------------------------------------------------
class TestMakeFactory:
    """Tests for the make() factory function."""

    def test_msp_aliases(self, simple_model):
        """All MSP aliases produce SoftmaxThreshold."""
        for name in ["msp", "softmax", "threshold"]:
            m = make(name, simple_model)
            assert isinstance(m, SoftmaxThreshold)

    def test_selectivenet_aliases(self):
        """All SelectiveNet aliases produce SelectiveNet."""
        backbone = nn.Linear(10, 32)
        for name in ["selectivenet", "sn"]:
            m = make(name, backbone, num_classes=5, num_features=32)
            assert isinstance(m, SelectiveNet)

    def test_gambler_aliases(self):
        """All DeepGambler aliases produce DeepGambler."""
        backbone = nn.Linear(10, 32)
        for name in ["gambler", "deepgambler"]:
            m = make(name, backbone, num_classes=5, num_features=32)
            assert isinstance(m, DeepGambler)

    def test_sat_aliases(self, simple_model, num_classes):
        """All SAT aliases produce SelfAdaptiveTraining."""
        for name in ["sat", "selfadaptive", "self-adaptive"]:
            m = make(name, simple_model, num_classes)
            assert isinstance(m, SelfAdaptiveTraining)

    def test_unknown_raises(self):
        """Unknown selector name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown selector"):
            make("not_a_real_method")

    def test_case_insensitive(self, simple_model):
        """Factory should be case-insensitive."""
        m = make("MSP", simple_model)
        assert isinstance(m, SoftmaxThreshold)


# ----------------------------------------------------------------------
#  Helper
# ----------------------------------------------------------------------
class TestInferOutputDim:
    """Tests for _infer_output_dim helper."""

    def test_restores_training_mode(self):
        """_infer_output_dim should restore the backbone's original training mode."""
        backbone = nn.Linear(10, 32)
        backbone.train()
        assert backbone.training is True

        dim = _infer_output_dim(backbone)
        assert dim == 32
        assert backbone.training is True  # restored

    def test_preserves_eval_mode(self):
        """Should leave eval-mode backbone in eval mode."""
        backbone = nn.Linear(10, 32)
        backbone.eval()
        _infer_output_dim(backbone)
        assert backbone.training is False
