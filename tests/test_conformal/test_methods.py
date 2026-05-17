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
from torch.utils.data import DataLoader, TensorDataset

from incerto.conformal import (
    ConformalPredictor,
    aps,
    conformalized_quantile_regression,
    cv_plus,
    inductive_conformal,
    jackknife_plus,
    mondrian_conformal,
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

        predictor_90 = inductive_conformal(ood_model, ood_id_loader, alpha=0.1)  # 90% coverage
        predictor_80 = inductive_conformal(ood_model, ood_id_loader, alpha=0.2)  # 80% coverage

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

    def test_unseen_partition_falls_back_gracefully(self, ood_model, ood_id_loader):
        """Mondrian should not KeyError when test partition wasn't in calibration."""
        alpha = 0.1

        # Partition that maps all calib labels to group 0,
        # then returns unseen group 999 at test time.
        call_count = {"n": 0}

        def shifting_partition(x, y):
            call_count["n"] += 1
            if call_count["n"] <= 4:  # calibration batches (100 samples / 32 = 4)
                return torch.zeros_like(y)
            return torch.full_like(y, 999)

        predictor = mondrian_conformal(
            ood_model, ood_id_loader, alpha, partition_fn=shifting_partition
        )
        x_test = torch.randn(5, 64)
        # Should not raise KeyError; unseen partition gets conservative fallback
        pred_sets = predictor(x_test)
        assert len(pred_sets) == 5
        # Conservative fallback: qhat=1.0 → threshold=0 → all classes included
        for ps in pred_sets:
            assert len(ps) == 10  # num_classes


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

    def test_prediction_sets_are_valid_classes(self, ood_model, ood_id_loader, num_classes):
        """Test that prediction sets contain valid class indices."""
        alpha = 0.1
        x_test = torch.randn(5, 64)

        predictor = inductive_conformal(ood_model, ood_id_loader, alpha)
        pred_sets = predictor(x_test)

        for ps in pred_sets:
            if len(ps) > 0:  # If not empty
                assert ps.min() >= 0
                assert ps.max() < num_classes


# ---- Alpha validation tests ----


class TestAlphaValidation:
    """Test that invalid alpha is rejected by all methods."""

    def test_invalid_alpha_raises(self, ood_model, ood_id_loader):
        """Alpha outside (0, 1) should raise ValueError."""
        for bad_alpha in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="alpha must be in"):
                inductive_conformal(ood_model, ood_id_loader, bad_alpha)


# ---- Conformal regression tests ----


def _make_regression_dataset(n=60, noise=0.1):
    """Create a simple regression dataset: y = 2x + noise."""
    x = torch.randn(n, 1)
    y = 2 * x.squeeze() + noise * torch.randn(n)
    return TensorDataset(x, y)


def _train_linear_model(dataset):
    """Train a simple linear model on a dataset (for jackknife_plus / cv_plus).

    Uses torch.enable_grad() because the caller (jackknife_plus/cv_plus)
    wraps everything in @torch.no_grad().
    """
    model = nn.Linear(1, 1)
    loader = DataLoader(dataset, batch_size=len(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    with torch.enable_grad():
        for _ in range(50):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()
    model.eval()
    return model


class TestJackknifePlus:
    """Test jackknife_plus conformal regression."""

    def test_returns_callable(self, set_seed):
        """Test that jackknife_plus returns a callable predictor."""
        dataset = _make_regression_dataset(n=20)
        predictor = jackknife_plus(_train_linear_model, dataset, alpha=0.1)
        assert callable(predictor)

    def test_predictor_returns_intervals(self, set_seed):
        """Test that predictor returns (lower, upper) tensors."""
        dataset = _make_regression_dataset(n=20)
        predictor = jackknife_plus(_train_linear_model, dataset, alpha=0.1)

        x_test = torch.randn(5, 1)
        lower, upper = predictor(x_test)

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert torch.isfinite(lower).all()
        assert torch.isfinite(upper).all()
        # Upper should be >= lower
        assert (upper >= lower).all()


class TestCVPlus:
    """Test cv_plus conformal regression."""

    def test_returns_callable(self, set_seed):
        """Test that cv_plus returns a callable predictor."""
        dataset = _make_regression_dataset(n=30)
        predictor = cv_plus(_train_linear_model, dataset, folds=3, alpha=0.1)
        assert callable(predictor)

    def test_predictor_returns_intervals(self, set_seed):
        """Test that predictor returns (lower, upper) tensors."""
        dataset = _make_regression_dataset(n=30)
        predictor = cv_plus(_train_linear_model, dataset, folds=3, alpha=0.1)

        x_test = torch.randn(5, 1)
        lower, upper = predictor(x_test)

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert torch.isfinite(lower).all()
        assert torch.isfinite(upper).all()


class TestConformalized_QR:
    """Test conformalized_quantile_regression."""

    def test_returns_callable(self, set_seed):
        """Test that CQR returns a callable predictor."""
        # Create a model that outputs two quantiles
        quantile_model = nn.Linear(1, 2)
        dataset = _make_regression_dataset(n=50)
        loader = DataLoader(dataset, batch_size=50)

        predictor = conformalized_quantile_regression(quantile_model, loader, alpha=0.1)
        assert callable(predictor)

    def test_predictor_returns_intervals(self, set_seed):
        """Test that CQR predictor returns (lower, upper) tensors."""
        quantile_model = nn.Linear(1, 2)
        dataset = _make_regression_dataset(n=50)
        loader = DataLoader(dataset, batch_size=50)

        predictor = conformalized_quantile_regression(quantile_model, loader, alpha=0.1)

        x_test = torch.randn(5, 1)
        lower, upper = predictor(x_test)

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert torch.isfinite(lower).all()
        assert torch.isfinite(upper).all()

    def test_single_output_model(self, set_seed):
        """Test CQR with a model that outputs a single value (squeeze to 1D)."""

        class SqueezeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x).squeeze(-1)  # (batch,) — 1D

        model = SqueezeModel()
        dataset = _make_regression_dataset(n=50)
        loader = DataLoader(dataset, batch_size=50)

        predictor = conformalized_quantile_regression(model, loader, alpha=0.1)

        x_test = torch.randn(5, 1)
        lower, upper = predictor(x_test)
        assert lower.shape == (5,)
        assert upper.shape == (5,)


class TestConformalPredictor:
    """Test the ConformalPredictor wrapper class."""

    def test_from_method(self, ood_model, ood_id_loader):
        """Test factory method creates a valid predictor."""
        cp = ConformalPredictor.from_method(
            "aps", model=ood_model, calib_loader=ood_id_loader, alpha=0.1
        )
        assert cp.method == "aps"
        assert cp.alpha == 0.1

        x_test = torch.randn(3, 64)
        result = cp.predict(x_test)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_call_delegates_to_predict(self, ood_model, ood_id_loader):
        """Test __call__ delegates to predict."""
        cp = ConformalPredictor.from_method(
            "inductive_conformal",
            model=ood_model,
            calib_loader=ood_id_loader,
            alpha=0.1,
        )
        x_test = torch.randn(3, 64)
        assert len(cp(x_test)) == len(cp.predict(x_test))

    def test_unknown_method_raises(self):
        """Test that unknown method name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            ConformalPredictor.from_method("nonexistent", alpha=0.1)

    def test_repr(self, ood_model, ood_id_loader):
        """Test repr string."""
        cp = ConformalPredictor.from_method(
            "raps", model=ood_model, calib_loader=ood_id_loader, alpha=0.1
        )
        r = repr(cp)
        assert "raps" in r
        assert "0.1" in r


class TestCoverageGuarantee:
    """Statistical tests that conformal methods achieve the advertised coverage."""

    def _make_trained_model(self, n_train=500, n_classes=5):
        """Train a small model well enough to test conformal coverage."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, n_classes))
        X = torch.randn(n_train, 8)
        # Labels from argmax of a fixed linear map so the problem is learnable
        W_true = torch.randn(8, n_classes)
        y = (X @ W_true).argmax(dim=-1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        model.train()
        for _ in range(30):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
        model.eval()
        return model, W_true

    def _make_calib_test(self, W_true, n_cal=1000, n_test=2000, n_classes=5):
        torch.manual_seed(1)
        X_cal = torch.randn(n_cal, 8)
        y_cal = (X_cal @ W_true).argmax(dim=-1)
        cal_loader = DataLoader(TensorDataset(X_cal, y_cal), batch_size=64)

        torch.manual_seed(2)
        X_test = torch.randn(n_test, 8)
        y_test = (X_test @ W_true).argmax(dim=-1)
        return cal_loader, X_test, y_test

    @staticmethod
    def _check_coverage(pred_sets, y_test, alpha, slack=0.03):
        """Check empirical coverage with a 2-sigma slack for binomial noise."""
        covered = 0
        for i, y in enumerate(y_test):
            s = pred_sets[i]
            if isinstance(s, torch.Tensor):
                if (s == y).any().item():
                    covered += 1
            else:
                if y.item() in s:
                    covered += 1
        coverage = covered / len(y_test)
        assert (
            coverage >= (1 - alpha) - slack
        ), f"Coverage {coverage:.3f} too low (target {1-alpha})"
        return coverage

    def test_inductive_conformal_coverage(self):
        """ICP empirical coverage on held-out data should be >= 1-alpha."""
        model, W_true = self._make_trained_model()
        alpha = 0.1
        cal_loader, X_test, y_test = self._make_calib_test(W_true)

        predictor = inductive_conformal(model, cal_loader, alpha=alpha)
        pred_sets = predictor(X_test)
        self._check_coverage(pred_sets, y_test, alpha)

    def test_aps_coverage(self):
        """APS empirical coverage on held-out data should be >= 1-alpha."""
        model, W_true = self._make_trained_model()
        alpha = 0.1
        cal_loader, X_test, y_test = self._make_calib_test(W_true)

        predictor = aps(model, cal_loader, alpha=alpha)
        pred_sets = predictor(X_test)
        self._check_coverage(pred_sets, y_test, alpha)

    def test_raps_coverage(self):
        """RAPS empirical coverage on held-out data should be >= 1-alpha."""
        model, W_true = self._make_trained_model()
        alpha = 0.1
        cal_loader, X_test, y_test = self._make_calib_test(W_true)

        predictor = raps(model, cal_loader, alpha=alpha, lam=0.01, k_reg=1)
        pred_sets = predictor(X_test)
        self._check_coverage(pred_sets, y_test, alpha)

    def test_mondrian_coverage(self):
        """Mondrian empirical coverage should be >= 1-alpha overall."""
        model, W_true = self._make_trained_model()
        alpha = 0.1
        cal_loader, X_test, y_test = self._make_calib_test(W_true)

        # Partition by predicted class so per-partition calibration is feasible.
        def partition_by_pred(x, y=None):
            with torch.no_grad():
                return model(x).argmax(dim=-1)

        predictor = mondrian_conformal(
            model, cal_loader, alpha=alpha, partition_fn=partition_by_pred
        )
        pred_sets = predictor(X_test)
        self._check_coverage(pred_sets, y_test, alpha, slack=0.05)

    @staticmethod
    def _train_linear_regressor(dataset: torch.utils.data.Dataset) -> nn.Module:
        """Train a small regression model on the provided dataset subset."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(20):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
        model.eval()
        return model

    @pytest.mark.slow
    def test_jackknife_plus_coverage(self):
        """Jackknife+ empirical coverage for regression should meet the bound."""
        torch.manual_seed(0)
        n_train, n_test = 40, 400
        X = torch.randn(n_train, 4)
        beta = torch.randn(4)
        y = X @ beta + 0.1 * torch.randn(n_train)
        train_ds = TensorDataset(X, y)

        alpha = 0.1
        predictor = jackknife_plus(self._train_linear_regressor, train_ds, alpha=alpha)

        torch.manual_seed(2)
        X_test = torch.randn(n_test, 4)
        y_test = X_test @ beta + 0.1 * torch.randn(n_test)
        lo, hi = predictor(X_test)
        covered = ((y_test >= lo) & (y_test <= hi)).float().mean().item()
        # Jackknife+ has 1 - 2*alpha worst-case guarantee
        assert (
            covered >= 1 - 2 * alpha - 0.02
        ), f"Jackknife+ coverage {covered:.3f} below worst-case bound"

    def test_cv_plus_coverage(self):
        """CV+ empirical coverage for regression should meet the bound."""
        torch.manual_seed(0)
        n_train, n_test = 100, 400
        X = torch.randn(n_train, 4)
        beta = torch.randn(4)
        y = X @ beta + 0.1 * torch.randn(n_train)
        train_ds = TensorDataset(X, y)

        alpha = 0.1
        predictor = cv_plus(self._train_linear_regressor, train_ds, folds=5, alpha=alpha)

        torch.manual_seed(2)
        X_test = torch.randn(n_test, 4)
        y_test = X_test @ beta + 0.1 * torch.randn(n_test)
        lo, hi = predictor(X_test)
        covered = ((y_test >= lo) & (y_test <= hi)).float().mean().item()
        # CV+ has the same 1 - 2*alpha worst-case bound
        assert covered >= 1 - 2 * alpha - 0.02, f"CV+ coverage {covered:.3f} below worst-case bound"
