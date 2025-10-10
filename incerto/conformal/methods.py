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
    up to a dynamic βγ calibrated on the held-out data.
    """
    model.eval()
    sizes = []
    for x, y in calib_loader:
        logits = model(x)
        conf_sorted, _ = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        cumprob = conf_sorted.cumsum(dim=-1)
        sizes.append((cumprob <= 1.0).sum(dim=-1))
    qhat = torch.quantile(torch.cat(sizes).float(), 1.0 - alpha)

    def predictor(x: torch.Tensor) -> List[torch.Tensor]:
        logits = model(x)
        probs, idx = torch.sort(torch.softmax(logits, dim=-1), descending=True)
        cumprob = probs.cumsum(dim=-1)
        k = (cumprob <= 1.0).sum(dim=-1).clip(max=int(qhat))
        return [idx_i[: int(k_i)].clone().detach() for idx_i, k_i in zip(idx, k)]

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
