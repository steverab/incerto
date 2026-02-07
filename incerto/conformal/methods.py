"""
incerto.conformal.methods
-------------------------
Stateless helper functions that wrap a *trained* base model and produce
prediction sets or intervals at a user-specified mis-coverage rate α.

Each method adds only what is necessary for conformal inference—no
optimisers, schedulers, or training loops are defined here.
"""

from __future__ import annotations
from typing import Callable, Tuple, List, Union
import math
import torch


def _validate_alpha(alpha: float) -> None:
    """Validate that alpha is in (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _conformal_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute the conformal quantile with exact finite-sample correction.

    Returns the ⌈(1 − α)(n + 1)⌉-th smallest score, which guarantees
    1 − α coverage under exchangeability.
    """
    n = len(scores)
    sorted_scores = torch.sort(scores)[0]
    k = math.ceil((1 - alpha) * (n + 1))  # 1-indexed
    k = max(1, min(k, n))
    return sorted_scores[k - 1]


class ConformalPredictor:
    """Thin wrapper around a calibrated conformal predictor.

    Provides a consistent object-oriented interface for both classification
    (prediction sets) and regression (intervals) conformal methods.

    Args:
        predictor: Callable returned by a conformal method function.
        method: Name of the conformal method used.
        alpha: Miscoverage rate used during calibration.

    Example:
        >>> cp = ConformalPredictor.from_method(
        ...     "raps", model=model, calib_loader=loader, alpha=0.1
        ... )
        >>> pred_sets = cp.predict(x_test)
    """

    def __init__(
        self,
        predictor: Callable,
        method: str = "unknown",
        alpha: float = 0.0,
    ):
        self._predictor = predictor
        self.method = method
        self.alpha = alpha

    def predict(
        self, x: torch.Tensor
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Run the calibrated predictor on new inputs.

        For classification: returns ``List[torch.Tensor]`` of prediction sets.
        For regression: returns ``(lower, upper)`` interval bounds.
        """
        return self._predictor(x)

    def __call__(
        self, x: torch.Tensor
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.predict(x)

    def __repr__(self) -> str:
        return f"ConformalPredictor(method='{self.method}', alpha={self.alpha})"

    @classmethod
    def from_method(cls, method: str, **kwargs) -> "ConformalPredictor":
        """Convenience factory that calls the named method and wraps the result.

        Args:
            method: One of ``'inductive_conformal'``, ``'mondrian_conformal'``,
                    ``'aps'``, ``'raps'``, ``'jackknife_plus'``, ``'cv_plus'``,
                    ``'conformalized_quantile_regression'``.
            **kwargs: Arguments forwarded to the method function.

        Returns:
            A :class:`ConformalPredictor` wrapping the calibrated predictor.
        """
        methods = {
            "inductive_conformal": inductive_conformal,
            "mondrian_conformal": mondrian_conformal,
            "aps": aps,
            "raps": raps,
            "jackknife_plus": jackknife_plus,
            "cv_plus": cv_plus,
            "conformalized_quantile_regression": conformalized_quantile_regression,
        }
        if method not in methods:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods)}")
        alpha = kwargs.get("alpha", 0.0)
        predictor = methods[method](**kwargs)
        return cls(predictor, method=method, alpha=alpha)


@torch.no_grad()
def inductive_conformal(
    model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """
    Classical Inductive Conformal Prediction (ICP)
    — Vovk, Gammerman, and Shafer, *Algorithmic Learning in a Random World* (2005).

    Returns a predictor f̂(x) that outputs a prediction set for classification.
    """
    _validate_alpha(alpha)
    model.eval()
    scores = []
    for x, y in calib_loader:
        logits = model(x)
        conf = torch.softmax(logits, dim=-1)
        # conformity score: 1 − probability assigned to the true class
        scores.append(1.0 - conf[torch.arange(len(y)), y])
    all_scores = torch.cat(scores)
    qhat = _conformal_quantile(all_scores, alpha)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            conf = torch.softmax(logits, dim=-1)
            return [(conf_i >= 1.0 - qhat).nonzero().squeeze(-1) for conf_i in conf]

    return predictor


@torch.no_grad()
def mondrian_conformal(
    model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
    partition_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """
    Mondrian Conformal Prediction
    — Papadopoulos, *Reliable Classification with Conformal Predictors* (2008).

    Allows per-class (or arbitrary partition) calibration to guarantee *conditional*
    coverage within each partition cell.
    """
    _validate_alpha(alpha)
    if partition_fn is None:
        # default: partition by true label
        partition_fn = lambda x, y: y  # noqa: E731

    model.eval()
    parts = {}
    for x, y in calib_loader:
        logits = model(x)
        conf = torch.softmax(logits, dim=-1)
        part = partition_fn(x, y)
        for p, c, yy in zip(part, conf, y):
            scores = parts.setdefault(int(p), [])
            scores.append(1.0 - c[yy])
    qhats = {k: _conformal_quantile(torch.tensor(v), alpha) for k, v in parts.items()}

    # Conservative fallback for partitions unseen during calibration:
    # qhat=1.0 means threshold 1−1=0, so every class is included.
    _default_qhat = torch.tensor(1.0)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            conf = torch.softmax(logits, dim=-1)
            part = partition_fn(x, logits.argmax(-1))
            return [
                (ci >= 1.0 - qhats.get(int(pi), _default_qhat)).nonzero().squeeze(-1)
                for ci, pi in zip(conf, part)
            ]

    return predictor


@torch.no_grad()
def aps(
    model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """
    Adaptive Prediction Sets (APS)
    — Romano, Patterson, and Candes, *NeurIPS 2020*.

    Produces variable-sized sets by thresholding cumulative probability mass
    up to and including the true label, calibrated on held-out data.
    """
    _validate_alpha(alpha)
    model.eval()
    scores = []
    for x, y in calib_loader:
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        probs_sorted, idx_sorted = torch.sort(probs, descending=True, dim=-1)
        cumprobs = probs_sorted.cumsum(dim=-1)

        # Find where true label appears in sorted order
        ranks = (idx_sorted == y.unsqueeze(-1)).nonzero(as_tuple=True)[1]
        # Score is cumulative probability up to and including true label
        scores.append(cumprobs[torch.arange(len(y)), ranks])

    all_scores = torch.cat(scores)
    qhat = _conformal_quantile(all_scores, alpha)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs, idx = torch.sort(
                torch.softmax(logits, dim=-1), descending=True, dim=-1
            )
            cumprobs = probs.cumsum(dim=-1)
            # Include classes until cumulative probability exceeds threshold
            sets = []
            for idx_i, cumprobs_i in zip(idx, cumprobs):
                s = idx_i[cumprobs_i <= qhat]
                if len(s) == 0:
                    s = idx_i[:1]  # always include the most probable class
                sets.append(s)
            return sets

    return predictor


@torch.no_grad()
def raps(
    model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
    lam: float = 0.0,
    k_reg: int = 1,
) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """
    Regularized APS (RAPS)
    — Angelopoulos, Bates, Malik, and Jordan, *ICLR 2021*.

    Adds ℓ₁ regularisation (λ) beyond rank k_reg and minimum size constraint to APS.
    """
    _validate_alpha(alpha)
    model.eval()
    # compute calibration scores following RAPS definition
    scores = []
    for x, y in calib_loader:
        logits = model(x)
        probs, idx = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        rank = (idx == y[:, None]).nonzero()[:, 1]
        ranks = torch.arange(1, probs.size(-1) + 1, device=probs.device)
        penalty = lam * torch.clamp(ranks - k_reg, min=0)
        g = probs.cumsum(dim=-1) + penalty
        scores.append(g[torch.arange(len(y)), rank])
    all_scores = torch.cat(scores)
    qhat = _conformal_quantile(all_scores, alpha)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs, idx = torch.sort(torch.softmax(logits, dim=-1), descending=True)
            ranks = torch.arange(1, probs.size(-1) + 1, device=probs.device)
            penalty = lam * torch.clamp(ranks - k_reg, min=0)
            g = probs.cumsum(dim=-1) + penalty
            S = (g <= qhat).long()
            # enforce minimum size k_reg
            mask_min = torch.arange(probs.size(-1), device=probs.device)[None] < k_reg
            S = S | mask_min
            return [(idx_i[S_i == 1]).clone().detach() for idx_i, S_i in zip(idx, S)]

    return predictor


# -------- Regression flavours (absolute residual conformity) -------- #


def jackknife_plus(
    model_fn: Callable[[torch.utils.data.Dataset], torch.nn.Module],
    train_dataset: torch.utils.data.Dataset,
    alpha: float,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Jackknife+ Intervals
    — Barber, Candès, and Ramdas, *Ann. Stat.* 2021.

    Produces prediction intervals using leave-one-out cross-validation with
    the Jackknife+ aggregation rule.  All n LOO models are retained and used
    at prediction time to construct valid intervals.

    Args:
        model_fn: Function that trains a model on a provided dataset subset.
        train_dataset: Training dataset used for LOO calibration.
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage).
    """
    _validate_alpha(alpha)
    n = len(train_dataset)
    loo_models: List[torch.nn.Module] = []
    residuals = torch.empty(n)

    for i in range(n):
        leave_one_out = torch.utils.data.Subset(
            train_dataset, [j for j in range(n) if j != i]
        )
        model = model_fn(leave_one_out)
        model.eval()
        loo_models.append(model)
        xi, yi = train_dataset[i]
        with torch.no_grad():
            pred = model(xi.unsqueeze(0)).squeeze().cpu()
        residuals[i] = torch.abs(pred - yi)

    def predictor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loo_preds = []
        for i in range(n):
            loo_models[i].eval()
            with torch.no_grad():
                loo_preds.append(loo_models[i](x).squeeze())
        loo_preds = torch.stack(loo_preds, dim=0)  # (n, *test_shape)

        r = residuals
        if loo_preds.dim() > 1:
            r = residuals.unsqueeze(-1)

        lower_vals = loo_preds - r
        upper_vals = loo_preds + r

        # Jackknife+ quantiles (Barber et al., 2021, Theorem 1)
        # Lower: ⌈α(n+1)⌉-th smallest of {-∞} ∪ lower_vals
        # Upper: ⌈(1-α)(n+1)⌉-th smallest of upper_vals ∪ {+∞}
        lower_sorted = torch.sort(lower_vals, dim=0)[0]
        upper_sorted = torch.sort(upper_vals, dim=0)[0]

        k_lo = math.ceil(alpha * (n + 1)) - 2  # 0-indexed into actual values
        if k_lo < 0:
            lower = torch.full_like(lower_sorted[0], float("-inf"))
        else:
            lower = lower_sorted[k_lo]

        k_hi = math.ceil((1 - alpha) * (n + 1)) - 1  # 0-indexed
        if k_hi >= n:
            upper = torch.full_like(upper_sorted[0], float("inf"))
        else:
            upper = upper_sorted[k_hi]

        return lower, upper

    return predictor


def cv_plus(
    model_fn: Callable[[torch.utils.data.Dataset], torch.nn.Module],
    train_dataset: torch.utils.data.Dataset,
    folds: int,
    alpha: float,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Cross-Validation+ Intervals (CV+)
    — Barber et al., *Ann. Stat.* 2021.

    Uses k-fold cross-validation to produce prediction intervals.  All K
    fold models are retained and used at prediction time via the CV+
    aggregation rule.

    Args:
        model_fn: Function that trains a model on a provided dataset subset.
        train_dataset: Training dataset used for k-fold calibration.
        folds: Number of cross-validation folds.
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage).
    """
    _validate_alpha(alpha)
    n = len(train_dataset)
    idx = torch.randperm(n)
    fold_sizes = [(n + i) // folds for i in range(folds)]

    fold_models: List[torch.nn.Module] = []
    residuals = torch.empty(n)
    sample_to_fold = torch.empty(n, dtype=torch.long)

    offset = 0
    for k in range(folds):
        val_idx = idx[offset : offset + fold_sizes[k]]
        offset += fold_sizes[k]
        val_set = set(val_idx.tolist())
        train_idx = [i for i in idx.tolist() if i not in val_set]

        model = model_fn(torch.utils.data.Subset(train_dataset, train_idx))
        model.eval()
        fold_models.append(model)

        with torch.no_grad():
            Xk = torch.stack([train_dataset[i][0] for i in val_idx])
            yk = torch.tensor([train_dataset[i][1] for i in val_idx])
            preds = model(Xk).squeeze()
            fold_residuals = torch.abs(yk - preds)

        for j, vi in enumerate(val_idx.tolist()):
            residuals[vi] = fold_residuals[j]
            sample_to_fold[vi] = k

    def predictor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fold_preds = []
        for k in range(folds):
            fold_models[k].eval()
            with torch.no_grad():
                fold_preds.append(fold_models[k](x).squeeze())
        fold_preds = torch.stack(fold_preds, dim=0)  # (K, *test_shape)

        # Select the fold model prediction for each calibration sample
        selected = fold_preds[sample_to_fold]  # (n, *test_shape)

        r = residuals
        if selected.dim() > 1:
            r = residuals.unsqueeze(-1)

        lower_vals = selected - r
        upper_vals = selected + r

        # CV+ quantiles (Barber et al., 2021)
        lower_sorted = torch.sort(lower_vals, dim=0)[0]
        upper_sorted = torch.sort(upper_vals, dim=0)[0]

        k_lo = math.ceil(alpha * (n + 1)) - 2
        if k_lo < 0:
            lower = torch.full_like(lower_sorted[0], float("-inf"))
        else:
            lower = lower_sorted[k_lo]

        k_hi = math.ceil((1 - alpha) * (n + 1)) - 1
        if k_hi >= n:
            upper = torch.full_like(upper_sorted[0], float("inf"))
        else:
            upper = upper_sorted[k_hi]

        return lower, upper

    return predictor


@torch.no_grad()
def conformalized_quantile_regression(
    quantile_model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Conformalized Quantile Regression (CQR)
    — Romano, Patterson, and Candes, *NeurIPS 2019*.

    Combines quantile regression with split conformal prediction to produce
    adaptive prediction intervals with finite-sample coverage guarantees.

    The method works by:
    1. Training a quantile regression model to predict lower and upper quantiles
    2. Computing conformity scores on a calibration set as the maximum violation
    3. Using these scores to adjust the quantile predictions

    Args:
        quantile_model: A model that outputs (lower_quantile, upper_quantile) predictions.
                       Expected to output shape (batch_size, 2) where:
                       - [:, 0] is the lower quantile prediction
                       - [:, 1] is the upper quantile prediction
        calib_loader: DataLoader for calibration data
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)

    Returns:
        predictor: Function mapping inputs to (lower, upper) prediction intervals

    Reference:
        Romano, Patterson, and Candes. "Conformalized Quantile Regression."
        NeurIPS 2019. https://arxiv.org/abs/1905.03222

    Example:
        >>> # Train quantile model (outputs lower and upper quantiles)
        >>> quantile_net = QuantileRegressionNet()
        >>> # ... train quantile_net ...
        >>> predictor = conformalized_quantile_regression(
        ...     quantile_net, calib_loader, alpha=0.1
        ... )
        >>> lower, upper = predictor(test_x)
    """
    _validate_alpha(alpha)
    quantile_model.eval()
    scores = []

    for x, y in calib_loader:
        # Get quantile predictions (batch_size, 2)
        preds = quantile_model(x)

        if preds.dim() == 1:
            # If model outputs single value, assume it's the median
            # and we'll use symmetric intervals
            preds = preds.unsqueeze(-1).repeat(1, 2)

        q_lo = preds[:, 0]  # lower quantile prediction
        q_hi = preds[:, 1]  # upper quantile prediction

        # Conformity score: max of how much y exceeds the interval
        # score = max(q_lo - y, y - q_hi)
        score_low = q_lo - y
        score_high = y - q_hi
        score = torch.max(score_low, score_high)
        scores.append(score)

    # Compute calibrated quantile with finite-sample correction
    all_scores = torch.cat(scores)
    qhat = _conformal_quantile(all_scores, alpha)

    def predictor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantile_model.eval()
        with torch.no_grad():
            preds = quantile_model(x)

            if preds.dim() == 1:
                preds = preds.unsqueeze(-1).repeat(1, 2)

            # Adjust quantile predictions by calibrated correction
            lower = preds[:, 0] - qhat
            upper = preds[:, 1] + qhat

            return lower.squeeze(), upper.squeeze()

    return predictor
