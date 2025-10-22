"""
incerto.shift_detection.methods
===============================

Fast, sklearn-style wrappers around common shift-detection techniques.
Each detector exposes two methods:

    .fit(reference_loader)           # builds any reference statistics
    .score(test_loader) -> float     # returns a scalar shift score
"""

from __future__ import annotations
from typing import Optional

import torch
from torch.utils.data import DataLoader
from scipy import stats
from incerto.core.utils import pairwise_squared_euclidean

from .base import BaseShiftDetector
from . import metrics


# ------------------------------------------------------------------------- #
#   Non-parametric two-sample tests
# ------------------------------------------------------------------------- #
class MMDShiftDetector(BaseShiftDetector):
    r"""Kernel Maximum Mean Discrepancy (unbiased, Gaussian kernel).

    * Gretton et al., 2012
    """

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def _rbf(self, x, y):
        return torch.exp(-pairwise_squared_euclidean(x, y) / (2 * self.sigma**2))

    def _compute(self, test: torch.Tensor) -> float:
        x, y = self._reference, test
        k_xx = self._rbf(x, x).mean()
        k_yy = self._rbf(y, y).mean()
        k_xy = self._rbf(x, y).mean()
        return (k_xx + k_yy - 2 * k_xy).item()

    def state_dict(self) -> dict:
        """Save MMD detector state."""
        state = super().state_dict()
        state["sigma"] = self.sigma
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load MMD detector state."""
        super().load_state_dict(state)
        self.sigma = state["sigma"]

    def __repr__(self) -> str:
        fitted = hasattr(self, "_reference")
        n_samples = len(self._reference) if fitted else "not fitted"
        return f"MMDShiftDetector(sigma={self.sigma}, n_ref_samples={n_samples})"


class EnergyShiftDetector(BaseShiftDetector):
    """Energy distance – Szekely & Rizzo, 2013."""

    def _compute(self, test: torch.Tensor) -> float:
        x, y = self._reference, test
        return metrics.energy_distance(x, y)  # re-use metric

    def __repr__(self) -> str:
        fitted = hasattr(self, "_reference")
        n_samples = len(self._reference) if fitted else "not fitted"
        return f"EnergyShiftDetector(n_ref_samples={n_samples})"


class KSShiftDetector(BaseShiftDetector):
    """One-dimensional Kolmogorov–Smirnov test (per feature, max statistic)."""

    def _compute(self, test: torch.Tensor) -> float:
        return max(
            stats.ks_2samp(x.cpu().numpy(), test[:, i].cpu().numpy()).statistic
            for i, x in enumerate(self._reference.T)
        )

    def __repr__(self) -> str:
        fitted = hasattr(self, "_reference")
        if fitted:
            n_samples = len(self._reference)
            n_features = self._reference.shape[1] if self._reference.dim() > 1 else 1
            return (
                f"KSShiftDetector(n_ref_samples={n_samples}, n_features={n_features})"
            )
        return "KSShiftDetector(not fitted)"


# ------------------------------------------------------------------------- #
#   Black-box shift detectors (BBSD, classifier-based)
# ------------------------------------------------------------------------- #
class ClassifierShiftDetector(BaseShiftDetector):
    r"""Train a logistic regression to separate reference and test.

    * Lipton et al., 2018 (Black Box Shift Detection)
    """

    def __init__(self, clf_factory=None, device: Optional[str] = None) -> None:
        from sklearn.linear_model import LogisticRegression

        self.clf = clf_factory() if clf_factory else LogisticRegression(max_iter=1000)
        self.device = device

    def _compute(self, test: torch.Tensor) -> float:
        import numpy as np

        X_ref = self._reference.cpu().numpy()
        X_test = test.cpu().numpy()
        X = np.concatenate([X_ref, X_test], axis=0)
        y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_test))])
        self.clf.fit(X, y)
        proba = self.clf.predict_proba(X_test)[:, 1]
        # Mean output probability should be ~0.5 under no shift
        return abs(proba.mean() - 0.5) * 2

    def state_dict(self) -> dict:
        """Save classifier shift detector state."""
        import pickle

        state = super().state_dict()
        state["clf"] = pickle.dumps(self.clf)
        state["device"] = self.device
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load classifier shift detector state."""
        import pickle

        super().load_state_dict(state)
        self.clf = pickle.loads(state["clf"])
        self.device = state["device"]

    def __repr__(self) -> str:
        fitted = hasattr(self, "_reference")
        n_samples = len(self._reference) if fitted else "not fitted"
        return f"ClassifierShiftDetector(n_ref_samples={n_samples})"


# Alias for BBSD
BBSDDetector = ClassifierShiftDetector


class LabelShiftDetector:
    """
    Black-Box Shift Detection for label shift.

    Detects and quantifies label shift (prior probability shift) using
    confusion matrix estimation.

    Reference:
        Lipton et al., "Detecting and Correcting for Label Shift with Black Box Predictors" (ICML 2018)

    Args:
        num_classes: Number of classes
        calibrated: Whether predictions are calibrated
    """

    def __init__(self, num_classes: int, calibrated: bool = False):
        self.num_classes = num_classes
        self.calibrated = calibrated
        self.source_label_dist = None
        self.confusion_matrix = None

    def fit(
        self,
        model: torch.nn.Module,
        source_loader: DataLoader,
        validation_loader: DataLoader,
    ):
        """
        Fit label shift detector.

        Args:
            model: Trained classifier
            source_loader: Source domain data with labels
            validation_loader: Validation set from source domain
        """
        import numpy as np

        model.eval()

        # Estimate source label distribution
        all_labels = []
        for _, y in source_loader:
            all_labels.append(y)
        all_labels = torch.cat(all_labels)

        label_counts = torch.bincount(all_labels, minlength=self.num_classes)
        self.source_label_dist = label_counts.float() / label_counts.sum()

        # Estimate confusion matrix on validation set
        confusion = torch.zeros(self.num_classes, self.num_classes)

        with torch.no_grad():
            for x, y in validation_loader:
                outputs = model(x)
                preds = outputs.argmax(dim=-1)

                for true_label in range(self.num_classes):
                    mask = y == true_label
                    if mask.sum() > 0:
                        pred_counts = torch.bincount(
                            preds[mask], minlength=self.num_classes
                        )
                        confusion[true_label] += pred_counts.float()

        # Normalize rows (C[i,j] = P(pred=j | true=i))
        row_sums = confusion.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.confusion_matrix = confusion / row_sums

    def estimate_target_distribution(
        self,
        model: torch.nn.Module,
        target_loader: DataLoader,
    ) -> torch.Tensor:
        """
        Estimate target label distribution.

        Args:
            model: Trained classifier
            target_loader: Target domain data (no labels needed)

        Returns:
            Estimated target label distribution
        """
        if self.confusion_matrix is None:
            raise RuntimeError("Must call fit() first")

        model.eval()

        # Get predictions on target data
        all_preds = []
        with torch.no_grad():
            for x, _ in target_loader:
                outputs = model(x)
                preds = outputs.argmax(dim=-1)
                all_preds.append(preds)

        all_preds = torch.cat(all_preds)

        # Compute empirical prediction distribution
        pred_counts = torch.bincount(all_preds, minlength=self.num_classes)
        pred_dist = pred_counts.float() / pred_counts.sum()

        # Solve: pred_dist = C^T @ target_dist
        # target_dist = (C^T)^{-1} @ pred_dist
        try:
            target_dist = torch.linalg.solve(self.confusion_matrix.T, pred_dist)

            # Project to simplex (ensure non-negative and sum to 1)
            target_dist = torch.clamp(target_dist, min=0)
            target_dist = target_dist / target_dist.sum()

        except:
            # Fallback to least squares if matrix is singular
            target_dist = torch.linalg.lstsq(
                self.confusion_matrix.T, pred_dist
            ).solution
            target_dist = torch.clamp(target_dist, min=0)
            target_dist = target_dist / target_dist.sum()

        return target_dist

    def compute_shift_magnitude(
        self,
        model: torch.nn.Module,
        target_loader: DataLoader,
        metric: str = "tvd",
    ) -> float:
        """
        Compute magnitude of label shift.

        Args:
            model: Trained classifier
            target_loader: Target domain data
            metric: Metric to use ('tvd', 'kl', 'l2')

        Returns:
            Shift magnitude
        """
        target_dist = self.estimate_target_distribution(model, target_loader)

        if metric == "tvd":
            # Total variation distance
            return 0.5 * torch.abs(target_dist - self.source_label_dist).sum().item()
        elif metric == "kl":
            # KL divergence
            return (
                (
                    target_dist
                    * torch.log(
                        (target_dist + 1e-10) / (self.source_label_dist + 1e-10)
                    )
                )
                .sum()
                .item()
            )
        elif metric == "l2":
            # L2 distance
            return torch.norm(target_dist - self.source_label_dist, p=2).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def state_dict(self) -> dict:
        """Save label shift detector state."""
        return {
            "num_classes": self.num_classes,
            "calibrated": self.calibrated,
            "source_label_dist": self.source_label_dist,
            "confusion_matrix": self.confusion_matrix,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load label shift detector state."""
        from ..exceptions import SerializationError

        try:
            self.num_classes = state["num_classes"]
            self.calibrated = state["calibrated"]
            self.source_label_dist = state["source_label_dist"]
            self.confusion_matrix = state["confusion_matrix"]
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def save(self, path: str) -> None:
        """Save label shift detector state."""
        from ..exceptions import SerializationError

        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise SerializationError(f"Failed to save to {path}: {e}") from e

    @classmethod
    def load(
        cls, path: str, num_classes: int, calibrated: bool = False
    ) -> "LabelShiftDetector":
        """Load label shift detector from file."""
        from ..exceptions import SerializationError

        try:
            detector = cls(num_classes, calibrated)
            state = torch.load(path)
            detector.load_state_dict(state)
            return detector
        except Exception as e:
            raise SerializationError(f"Failed to load from {path}: {e}") from e

    def __repr__(self) -> str:
        fitted = self.source_label_dist is not None
        return f"LabelShiftDetector(num_classes={self.num_classes}, calibrated={self.calibrated}, fitted={fitted})"


class ImportanceWeightingShift:
    """
    Importance weighting for covariate shift adaptation.

    Estimates density ratio w(x) = p_target(x) / p_source(x) and
    uses it to re-weight training samples.

    Reference:
        Sugiyama et al., "Direct Importance Estimation with Model Selection" (NIPS 2007)

    Args:
        method: Estimation method ('kernel', 'logistic', 'kliep')
        alpha: Regularization parameter
    """

    def __init__(self, method: str = "logistic", alpha: float = 0.01):
        self.method = method
        self.alpha = alpha
        self.weights_model = None

    def fit(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ):
        """
        Estimate importance weights.

        Args:
            source_features: Features from source domain (N_s, D)
            target_features: Features from target domain (N_t, D)
        """
        import numpy as np

        if self.method == "logistic":
            # Train logistic regression to discriminate source vs target
            from sklearn.linear_model import LogisticRegression

            X_source = source_features.cpu().numpy()
            X_target = target_features.cpu().numpy()

            X = np.concatenate([X_source, X_target], axis=0)
            y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])

            clf = LogisticRegression(C=1.0 / self.alpha, max_iter=1000)
            clf.fit(X, y)

            self.weights_model = clf

        elif self.method == "kernel":
            # Kernel mean matching (KMM)
            self._fit_kernel_weights(source_features, target_features)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_kernel_weights(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ):
        """Fit kernel-based importance weights."""
        # Simplified KMM implementation
        from sklearn.metrics.pairwise import rbf_kernel

        X_s = source_features.cpu().numpy()
        X_t = target_features.cpu().numpy()

        n_s = len(X_s)
        n_t = len(X_t)

        # Compute kernel matrices
        K = rbf_kernel(X_s, X_s)
        kappa = rbf_kernel(X_s, X_t).mean(axis=1)

        # Solve QP problem (simplified)
        # min 0.5 * w^T K w - kappa^T w
        # s.t. w >= 0, mean(w) = 1

        import numpy as np
        from scipy.optimize import minimize

        def objective(w):
            return 0.5 * w @ K @ w - kappa @ w + self.alpha * (w**2).sum()

        def constraint(w):
            return w.mean() - 1

        constraints = {"type": "eq", "fun": constraint}
        bounds = [(0, None) for _ in range(n_s)]

        result = minimize(
            objective,
            np.ones(n_s) / n_s,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        self.weights_model = torch.tensor(result.x, dtype=torch.float32)

    def compute_weights(
        self,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance weights for source samples.

        Args:
            source_features: Source domain features

        Returns:
            Importance weights
        """
        if self.weights_model is None:
            raise RuntimeError("Must call fit() first")

        if self.method == "logistic":
            # w(x) = P(target|x) / P(source|x)
            #      = p(x|target) / p(x|source)
            #      ~ P(target|x) / (1 - P(target|x))

            X = source_features.cpu().numpy()
            probs = self.weights_model.predict_proba(X)[:, 1]  # P(target|x)

            weights = probs / (1 - probs + 1e-10)
            weights = torch.tensor(weights, dtype=torch.float32)

            # Normalize weights to sum to n
            weights = weights / weights.mean()

            return weights

        elif self.method == "kernel":
            return self.weights_model

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def weighted_loss(
        self,
        loss: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply importance weights to loss.

        Args:
            loss: Per-sample losses
            weights: Importance weights

        Returns:
            Weighted average loss
        """
        return (loss * weights).mean()

    def state_dict(self) -> dict:
        """Save importance weighting state."""
        import pickle

        return {
            "method": self.method,
            "alpha": self.alpha,
            "weights_model": pickle.dumps(self.weights_model),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load importance weighting state."""
        import pickle
        from ..exceptions import SerializationError

        try:
            self.method = state["method"]
            self.alpha = state["alpha"]
            self.weights_model = pickle.loads(state["weights_model"])
        except Exception as e:
            raise SerializationError(f"Failed to load state: {e}") from e

    def save(self, path: str) -> None:
        """Save importance weighting state."""
        from ..exceptions import SerializationError

        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise SerializationError(f"Failed to save to {path}: {e}") from e

    @classmethod
    def load(
        cls, path: str, method: str = "logistic", alpha: float = 0.01
    ) -> "ImportanceWeightingShift":
        """Load importance weighting from file."""
        from ..exceptions import SerializationError

        try:
            instance = cls(method, alpha)
            state = torch.load(path)
            instance.load_state_dict(state)
            return instance
        except Exception as e:
            raise SerializationError(f"Failed to load from {path}: {e}") from e

    def __repr__(self) -> str:
        fitted = self.weights_model is not None
        return f"ImportanceWeightingShift(method='{self.method}', alpha={self.alpha}, fitted={fitted})"
