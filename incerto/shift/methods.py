"""
incerto.shift_detection.methods
===============================

Fast, sklearn-style wrappers around common shift-detection techniques.
Each detector exposes two methods:

    .fit(reference_loader)           # builds any reference statistics
    .score(test_loader) -> float     # returns a scalar shift score
"""

from __future__ import annotations
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader
from scipy import stats
from incerto.core.utils import pairwise_squared_euclidean  # hypothetical helper


class _BaseShiftDetector:
    """Shared machinery; child classes implement _compute()"""

    def fit(self, reference_loader: DataLoader) -> "Self":
        self._reference = torch.cat([x[0].detach() for x in reference_loader])
        return self

    @torch.no_grad()
    def score(self, test_loader: DataLoader) -> float:
        test_batch = torch.cat([x[0].detach() for x in test_loader])
        return self._compute(test_batch)

    # --------------------------------------------------------------------- #
    # subclasses override this
    # --------------------------------------------------------------------- #
    def _compute(self, test: torch.Tensor) -> float:
        raise NotImplementedError


# ------------------------------------------------------------------------- #
#   Non-parametric two-sample tests
# ------------------------------------------------------------------------- #
class MMDShiftDetector(_BaseShiftDetector):
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


class EnergyShiftDetector(_BaseShiftDetector):
    """Energy distance – Szekely & Rizzo, 2013."""

    def _compute(self, test: torch.Tensor) -> float:
        x, y = self._reference, test
        return metrics.energy_distance(x, y)  # re-use metric


class KSShiftDetector(_BaseShiftDetector):
    """One-dimensional Kolmogorov–Smirnov test (per feature, max statistic)."""

    def _compute(self, test: torch.Tensor) -> float:
        return max(
            stats.ks_2samp(x.cpu().numpy(), test[:, i].cpu().numpy()).statistic
            for i, x in enumerate(self._reference.T)
        )


# ------------------------------------------------------------------------- #
#   Black-box shift detectors (BBSD, classifier-based)
# ------------------------------------------------------------------------- #
class ClassifierShiftDetector(_BaseShiftDetector):
    r"""Train a logistic regression to separate reference and test.

    * Lipton et al., 2018 (Black Box Shift Detection)
    """

    def __init__(self, clf_factory, device: Optional[str] = None) -> None:
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
