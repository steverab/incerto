"""
incerto.conformal.methods
-------------------------
Stateless helper functions that wrap a *trained* base model and produce
prediction sets or intervals at a user-specified mis-coverage rate α.

Each method adds only what is necessary for conformal inference—no
optimisers, schedulers, or training loops are defined here.
"""

from __future__ import annotations
from typing import Callable, Tuple, List
import torch
import numpy as np

# type alias
Batch = Tuple[torch.Tensor, torch.Tensor]  # (inputs, labels)


@torch.no_grad()
def inductive_conformal(
    model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """
    Classical Inductive Conformal Prediction (ICP)
    — Vovk, Gammerman, and Shafer, *Algorithmic Learning in a Random World* (2005).

    Returns a predictor f̂(x) that outputs a prediction set (classification) or
    interval (regression) for any new x.
    """
    model.eval()
    scores = []
    for x, y in calib_loader:
        logits = model(x)
        conf = torch.softmax(logits, dim=-1)
        # conformity score: 1 − probability assigned to the true class
        scores.append(1.0 - conf[torch.arange(len(y)), y])
    qhat = torch.quantile(torch.cat(scores), 1.0 - alpha)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
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
    qhats = {k: torch.quantile(torch.tensor(v), 1.0 - alpha) for k, v in parts.items()}

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        logits = model(x)
        conf = torch.softmax(logits, dim=-1)
        part = partition_fn(x, logits.argmax(-1))
        return [
            (ci >= 1.0 - qhats[int(pi)]).nonzero().squeeze(-1)
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

    qhat = torch.quantile(torch.cat(scores), 1.0 - alpha)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        logits = model(x)
        probs, idx = torch.sort(torch.softmax(logits, dim=-1), descending=True, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        # Include classes until cumulative probability exceeds threshold
        return [
            (idx_i[cumprobs_i <= qhat]).clone().detach()
            for idx_i, cumprobs_i in zip(idx, cumprobs)
        ]

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
    — Tsesmelis et al., *ICML 2021*.

    Adds ℓ₁ regularisation (λ) and minimum size constraint (k_reg) to APS.
    """
    model.eval()
    # compute calibration scores following RAPS definition
    scores = []
    for x, y in calib_loader:
        logits = model(x)
        probs, idx = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        rank = (idx == y[:, None]).nonzero()[:, 1]
        g = probs.cumsum(dim=-1) + lam * torch.arange(
            1, probs.size(-1) + 1, device=probs.device
        )
        scores.append(g[torch.arange(len(y)), rank])
    qhat = torch.quantile(
        torch.cat(scores), (1.0 - alpha) * (1 + 1.0 / len(calib_loader.dataset))
    )

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        logits = model(x)
        probs, idx = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        g = probs.cumsum(dim=-1) + lam * torch.arange(
            1, probs.size(-1) + 1, device=probs.device
        )
        S = (g <= qhat).long()
        # enforce minimum size k_reg
        ks = torch.clamp(k_reg - S.sum(dim=-1), min=0)
        mask_extra = (
            torch.arange(probs.size(-1), device=probs.device)[None] < ks[:, None]
        )
        S = S | mask_extra
        return [(idx_i[S_i == 1]).clone().detach() for idx_i, S_i in zip(idx, S)]

    return predictor


# -------- Regression flavours (absolute residual conformity) -------- #


@torch.no_grad()
def jackknife_plus(
    model_fn: Callable[[torch.utils.data.Dataset], torch.nn.Module],
    train_dataset: torch.utils.data.Dataset,
    alpha: float,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Jackknife+ Intervals
    — Barber, Candès, and Ramdas, *Ann. Stat.* 2021.

    model_fn: function that re-trains a model on a supplied dataset split.
    """
    n = len(train_dataset)
    preds = torch.empty((n,))  # ŷ_i^(-i)
    for i in range(n):
        leave_one_out = torch.utils.data.Subset(
            train_dataset, [j for j in range(n) if j != i]
        )
        model = model_fn(leave_one_out)
        xi, yi = train_dataset[i]
        preds[i] = model(xi.unsqueeze(0)).squeeze().cpu()
    residuals = torch.abs(preds - torch.tensor([train_dataset[i][1] for i in range(n)]))
    q = torch.quantile(residuals, 1.0 - alpha)

    def predictor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        full_model = model_fn(train_dataset)
        mu = full_model(x).squeeze()
        return mu - q, mu + q

    return predictor


@torch.no_grad()
def cv_plus(
    model_fn: Callable[[torch.utils.data.Dataset], torch.nn.Module],
    train_dataset: torch.utils.data.Dataset,
    folds: int,
    alpha: float,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Cross-Validation+ Intervals (CV+)
    — Barber et al., *JASA* 2021.

    Offers less pessimistic intervals than Jackknife+ while controlling coverage.
    """
    # split indices
    n = len(train_dataset)
    idx = torch.randperm(n)
    fold_sizes = [(n + i) // folds for i in range(folds)]
    intervals = []
    for k in range(folds):
        val_idx = idx[sum(fold_sizes[:k]) : sum(fold_sizes[: k + 1])]
        train_idx = [i for i in idx if i not in val_idx]
        model = model_fn(torch.utils.data.Subset(train_dataset, train_idx))
        Xk = torch.stack([train_dataset[i][0] for i in val_idx])
        yk = torch.tensor([train_dataset[i][1] for i in val_idx])
        preds = model(Xk).squeeze()
        intervals.append((yk - preds, yk + preds))
    lo, hi = torch.cat([lo for lo, _ in intervals]), torch.cat(
        [hi for _, hi in intervals]
    )
    q_lo, q_hi = torch.quantile(lo, alpha / 2), torch.quantile(hi, 1 - alpha / 2)

    def predictor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # train on full data
        model = model_fn(train_dataset)
        mu = model(x).squeeze()
        return mu + q_lo, mu + q_hi

    return predictor


@torch.no_grad()
def conformalized_quantile_regression(
    quantile_model: torch.nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    alpha: float,
    q_low: float | None = None,
    q_high: float | None = None,
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
        q_low: Lower quantile level (default: alpha/2)
        q_high: Upper quantile level (default: 1 - alpha/2)

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
    if q_low is None:
        q_low = alpha / 2
    if q_high is None:
        q_high = 1 - alpha / 2

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
    n_calib = len(all_scores)
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_level = min(q_level, 1.0)  # ensure valid quantile

    qhat = torch.quantile(all_scores, q_level)

    def predictor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict conformalized intervals for new inputs.

        Args:
            x: Input tensor of shape (batch_size, ...)

        Returns:
            lower: Lower bound of prediction interval (batch_size,)
            upper: Upper bound of prediction interval (batch_size,)
        """
        quantile_model.eval()
        preds = quantile_model(x)

        if preds.dim() == 1:
            preds = preds.unsqueeze(-1).repeat(1, 2)

        # Adjust quantile predictions by calibrated correction
        lower = preds[:, 0] - qhat
        upper = preds[:, 1] + qhat

        return lower.squeeze(), upper.squeeze()

    return predictor
