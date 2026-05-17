"""
Training utilities for examples.

Common training loops and helpers to reduce boilerplate in examples.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str = "Training",
) -> dict:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        desc: Description for progress bar

    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": 100.0 * correct / total,
    }


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        desc: Description for progress bar

    Returns:
        Dictionary with 'loss', 'accuracy', 'logits', 'targets'
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_logits.append(outputs.cpu())
            all_targets.append(targets.cpu())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(data_loader),
        "accuracy": 100.0 * correct / total,
        "logits": all_logits,
        "targets": all_targets,
    }


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


__all__ = [
    "seed_everything",
    "train_epoch",
    "evaluate",
    "EarlyStopping",
]
