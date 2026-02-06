"""
Safe serialization helpers for sklearn models.

Replaces pickle-based serialization with plain-dict extraction of fitted
attributes, enabling torch.load(..., weights_only=True).
"""

from __future__ import annotations

import numpy as np


def serialize_isotonic(ir) -> dict:
    """Extract fitted IsotonicRegression attributes to a plain dict."""
    return {
        "X_thresholds_": ir.X_thresholds_.tolist(),
        "y_thresholds_": ir.y_thresholds_.tolist(),
        "X_min_": float(ir.X_min_),
        "X_max_": float(ir.X_max_),
        "increasing_": bool(ir.increasing_),
        "out_of_bounds": ir.out_of_bounds,
    }


def deserialize_isotonic(d: dict):
    """Reconstruct a fitted IsotonicRegression from a plain dict."""
    from scipy.interpolate import interp1d
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds=d["out_of_bounds"])
    ir.X_thresholds_ = np.array(d["X_thresholds_"])
    ir.y_thresholds_ = np.array(d["y_thresholds_"])
    ir.X_min_ = d["X_min_"]
    ir.X_max_ = d["X_max_"]
    ir.increasing_ = d["increasing_"]

    if len(ir.X_thresholds_) == 1:
        ir.f_ = lambda x, val=float(ir.y_thresholds_[0]): np.full_like(
            x, val, dtype=np.float64
        )
    else:
        ir.f_ = interp1d(
            ir.X_thresholds_,
            ir.y_thresholds_,
            kind="linear",
            bounds_error=False,
        )
    return ir


def serialize_logistic(lr) -> dict | None:
    """Extract fitted LogisticRegression attributes to a plain dict.

    Returns None if the model has not been fitted yet.
    """
    if not hasattr(lr, "coef_"):
        return {"_fitted": False, "C": lr.C, "max_iter": lr.max_iter}
    return {
        "_fitted": True,
        "coef_": lr.coef_.tolist(),
        "intercept_": lr.intercept_.tolist(),
        "classes_": lr.classes_.tolist(),
        "C": lr.C,
        "max_iter": lr.max_iter,
    }


def deserialize_logistic(d: dict):
    """Reconstruct a LogisticRegression from a plain dict."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=d.get("C", 1.0), max_iter=d.get("max_iter", 100))
    if d.get("_fitted", True):
        lr.coef_ = np.array(d["coef_"])
        lr.intercept_ = np.array(d["intercept_"])
        lr.classes_ = np.array(d["classes_"])
    return lr
