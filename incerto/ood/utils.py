"""
Utility functions for OOD detection methods.
"""

import torch
import numpy as np


def compute_threshold_at_tpr(
    id_scores: torch.Tensor | np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """
    Compute OOD score threshold that accepts ``target_tpr`` fraction of ID samples.

    The threshold is set at the ``target_tpr``-th percentile of the ID score
    distribution, so that ``target_tpr`` of in-distribution samples fall below
    it (i.e. are correctly classified as ID).

    Args:
        id_scores: Scores from in-distribution data (lower = more ID-like).
        target_tpr: Fraction of ID samples to accept (default: 0.95).

    Returns:
        Threshold value.
    """
    if isinstance(id_scores, torch.Tensor):
        id_scores = id_scores.cpu().numpy()

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

    Uses ``str.endswith`` matching on module names, consistent with the
    hook mechanism in :class:`Mahalanobis` and :class:`KNN`.

    Args:
        model: PyTorch model.
        data_loader: DataLoader containing input data.
        layer_name: Name (suffix) of layer to extract features from.

    Returns:
        Tensor of extracted features.
    """
    model.eval()
    features = []
    activation = {}

    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()

        return hook

    # Register hook (endswith matching, consistent with _FeatureHookMixin)
    handle = None
    matched_name = None
    for name, module in model.named_modules():
        if name.endswith(layer_name):
            handle = module.register_forward_hook(get_activation(name))
            matched_name = name
            break

    if handle is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    try:
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(next(model.parameters()).device)
                model(x)
                if matched_name in activation:
                    features.append(activation[matched_name].flatten(1).cpu())
    finally:
        handle.remove()

    return torch.cat(features, dim=0) if features else torch.tensor([])
