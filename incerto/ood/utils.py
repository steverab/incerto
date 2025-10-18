"""
Utility functions for OOD detection methods.
"""

import torch
import numpy as np


def compute_threshold_at_tpr(
    id_scores: torch.Tensor | np.ndarray,
    ood_scores: torch.Tensor | np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    Compute OOD threshold that achieves target TPR on ID data.

    Args:
        id_scores: Scores from in-distribution data (lower = more ID-like).
        ood_scores: Scores from out-of-distribution data (higher = more OOD-like).
        target_tpr: Target true positive rate (fraction of ID samples to accept).

    Returns:
        Threshold value.
    """
    if isinstance(id_scores, torch.Tensor):
        id_scores = id_scores.cpu().numpy()
    if isinstance(ood_scores, torch.Tensor):
        ood_scores = ood_scores.cpu().numpy()

    threshold = np.percentile(id_scores, target_tpr * 100)
    return float(threshold)


def get_ood_predictions(
    scores: torch.Tensor | np.ndarray, threshold: float
) -> np.ndarray:
    """
    Get binary OOD predictions based on threshold.

    Args:
        scores: OOD scores (higher = more OOD-like).
        threshold: Decision threshold.

    Returns:
        Binary array (1 = OOD, 0 = ID).
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    return (scores > threshold).astype(int)


def extract_features(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    layer_name: str = "penultimate",
) -> torch.Tensor:
    """
    Extract features from a specific layer of the model.

    Args:
        model: PyTorch model.
        data_loader: DataLoader containing input data.
        layer_name: Name of layer to extract features from.

    Returns:
        Tensor of extracted features.
    """
    model.eval()
    features = []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Register hook
    for name, module in model.named_modules():
        if layer_name in name:
            module.register_forward_hook(get_activation(name))
            break

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            model(x)
            if layer_name in activation:
                features.append(activation[layer_name].flatten(1).cpu())

    return torch.cat(features, dim=0) if features else torch.tensor([])
