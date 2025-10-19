"""
Comprehensive MNIST Example for the Incerto Library

This example demonstrates the full capabilities of the incerto library for uncertainty
quantification on the MNIST dataset, including:
- Calibration methods and visualizations
- Out-of-distribution detection
- Selective prediction
- Conformal prediction
- Distribution shift detection

Usage:
    python examples/mnist.py --epochs 5 --batch_size 128
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Calibration imports
from incerto.calibration.methods import (
    TemperatureScaling,
    VectorScaling,
    MatrixScaling,
    IdentityCalibrator,
)
from incerto.calibration.metrics import nll, brier_score, ece_score, mce_score
from incerto.calibration.visual import (
    plot_reliability_diagram,
    plot_confidence_histogram,
)

# OOD detection imports
from incerto.ood.methods import MSP, Energy, ODIN, MaxLogit
from incerto.ood.metrics import auroc, fpr_at_tpr, detection_accuracy
from incerto.ood.visual import plot_roc, score_hist

# Selective prediction imports
from incerto.sp.methods import SoftmaxThreshold, SelfAdaptiveTraining
from incerto.sp.metrics import coverage, risk, aurc
from incerto.sp.visual import plot_risk_coverage

# Conformal prediction imports
from incerto.conformal.methods import inductive_conformal, aps
from incerto.conformal.metrics import empirical_coverage, average_set_size


class MNISTModel(nn.Module):
    """Simple CNN for MNIST classification"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def get_features(self, x):
        """Extract features for OOD detection methods that need them"""
        return self.features(x)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device, return_data=False):
    """Evaluate model and optionally return logits and labels"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    logits_list, labels_list, features_list = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if return_data:
                logits_list.append(outputs.cpu())
                labels_list.append(targets.cpu())
                features_list.append(model.get_features(inputs).cpu())

            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()

    if return_data:
        return (
            total_loss / total,
            100.0 * correct / total,
            torch.cat(logits_list),
            torch.cat(labels_list),
            torch.cat(features_list),
        )
    return total_loss / total, 100.0 * correct / total


def demonstrate_calibration(model, cal_loader, test_loader, device, console, save_dir):
    """Demonstrate calibration methods and visualizations"""
    console.print("\n[bold cyan]=== CALIBRATION DEMONSTRATION ===[/]")

    # Get calibration and test data
    _, _, cal_logits, cal_labels, _ = evaluate(
        model, cal_loader, nn.CrossEntropyLoss(), device, return_data=True
    )
    _, _, test_logits, test_labels, _ = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, return_data=True
    )

    # Define calibration methods to test (MNIST has 10 classes)
    n_classes = 10
    calibrators = {
        "Uncalibrated": IdentityCalibrator(),
        "Temperature Scaling": TemperatureScaling(init_temp=1.0),
        "Vector Scaling": VectorScaling(n_classes=n_classes),
        "Matrix Scaling": MatrixScaling(n_classes=n_classes),
    }

    results = {}

    # Move calibrators to device if they are nn.Module-based
    for name in calibrators:
        if hasattr(calibrators[name], "to"):
            calibrators[name] = calibrators[name].to(device)

    # Fit and evaluate each calibrator
    for name, calibrator in calibrators.items():
        # Fit on calibration set (except identity)
        if name != "Uncalibrated":
            calibrator.fit(cal_logits.to(device), cal_labels.to(device))

        # Predict on test set
        cal_dist = calibrator.predict(test_logits.to(device))
        cal_probs = cal_dist.probs.cpu()
        cal_logits_adj = torch.log(cal_probs + 1e-12)

        # Compute metrics
        results[name] = {
            "NLL": nll(cal_logits_adj, test_labels),
            "Brier": brier_score(cal_logits_adj, test_labels),
            "ECE": ece_score(cal_logits_adj, test_labels),
            "MCE": mce_score(cal_logits_adj, test_labels),
        }

    # Display results table
    table = Table(title="Calibration Methods Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("NLL", style="magenta")
    table.add_column("Brier Score", style="magenta")
    table.add_column("ECE", style="magenta")
    table.add_column("MCE", style="magenta")

    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics['NLL']:.4f}",
            f"{metrics['Brier']:.4f}",
            f"{metrics['ECE']:.4f}",
            f"{metrics['MCE']:.4f}",
        )
    console.print(table)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for idx, (name, calibrator) in enumerate(calibrators.items()):
        ax = axes[idx // 2, idx % 2]

        # Get calibrated predictions
        cal_dist = calibrator.predict(test_logits.to(device))
        cal_probs = cal_dist.probs.cpu()
        cal_logits_adj = torch.log(cal_probs + 1e-12)

        # Plot reliability diagram
        plot_reliability_diagram(
            cal_logits_adj, test_labels, n_bins=10, ax=ax, title=name
        )

    plt.tight_layout()
    plt.savefig(save_dir / "calibration_reliability.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]Saved reliability diagrams to {save_dir / 'calibration_reliability.png'}[/]"
    )

    # Confidence histogram comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, calibrator) in enumerate(calibrators.items()):
        ax = axes[idx // 2, idx % 2]

        cal_dist = calibrator.predict(test_logits.to(device))
        cal_probs = cal_dist.probs.cpu()
        cal_logits_adj = torch.log(cal_probs + 1e-12)

        plot_confidence_histogram(cal_logits_adj, n_bins=10, ax=ax, title=name)

    plt.tight_layout()
    plt.savefig(save_dir / "calibration_confidence.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]Saved confidence histograms to {save_dir / 'calibration_confidence.png'}[/]"
    )

    return calibrators["Temperature Scaling"]  # Return best calibrator for later use


def demonstrate_ood_detection(
    model, test_loader, ood_loader, device, console, save_dir
):
    """Demonstrate out-of-distribution detection methods"""
    console.print("\n[bold cyan]=== OUT-OF-DISTRIBUTION DETECTION ===[/]")

    # Collect in-distribution and OOD samples (we need raw inputs for OOD detectors)
    model.eval()

    # Collect a subset of ID samples
    id_inputs_list = []
    for inputs, _ in test_loader:
        id_inputs_list.append(inputs)
        if len(id_inputs_list) >= 10:  # Limit to avoid memory issues
            break
    id_inputs = torch.cat(id_inputs_list).to(device)

    # Collect a subset of OOD samples
    ood_inputs_list = []
    for inputs, _ in ood_loader:
        ood_inputs_list.append(inputs)
        if len(ood_inputs_list) >= 10:
            break
    ood_inputs = torch.cat(ood_inputs_list).to(device)

    # Define OOD detection methods (they need the model)
    detectors = {
        "MSP": MSP(model),
        "Energy": Energy(model, temperature=1.0),
        "MaxLogit": MaxLogit(model),
        "ODIN": ODIN(model, temperature=1000.0, epsilon=0.0014),
    }

    results = {}

    # Evaluate each detector
    for name, detector in detectors.items():
        # Get scores for ID and OOD (higher score = more OOD-like)
        # Detach scores to remove gradients (needed for ODIN which uses input perturbations)
        id_scores = detector.score(id_inputs).detach()
        ood_scores = detector.score(ood_inputs).detach()

        # Compute metrics
        results[name] = {
            "AUROC": auroc(id_scores, ood_scores),
            "FPR@95": fpr_at_tpr(id_scores, ood_scores, tpr=0.95),
            "Detection Acc": detection_accuracy(id_scores, ood_scores),
        }

    # Display results table
    table = Table(title="OOD Detection Methods Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("AUROC", style="magenta")
    table.add_column("FPR@95%TPR", style="magenta")
    table.add_column("Detection Acc", style="magenta")

    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics['AUROC']:.4f}",
            f"{metrics['FPR@95']:.4f}",
            f"{metrics['Detection Acc']:.4f}",
        )
    console.print(table)

    # Create ROC curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, detector) in enumerate(detectors.items()):
        ax = axes[idx // 2, idx % 2]

        id_scores = detector.score(id_inputs).detach()
        ood_scores = detector.score(ood_inputs).detach()

        plot_roc(id_scores, ood_scores, ax=ax, label=name)
        ax.set_title(f"{name} (AUROC: {results[name]['AUROC']:.4f})")

    plt.tight_layout()
    plt.savefig(save_dir / "ood_roc_curves.png", dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved ROC curves to {save_dir / 'ood_roc_curves.png'}[/]")

    # Score histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, detector) in enumerate(detectors.items()):
        ax = axes[idx // 2, idx % 2]

        id_scores = detector.score(id_inputs).detach()
        ood_scores = detector.score(ood_inputs).detach()

        score_hist(id_scores, ood_scores, ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(save_dir / "ood_score_histograms.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]Saved score histograms to {save_dir / 'ood_score_histograms.png'}[/]"
    )


def demonstrate_selective_prediction(model, test_loader, device, console, save_dir):
    """Demonstrate selective prediction (rejection option)"""
    console.print("\n[bold cyan]=== SELECTIVE PREDICTION ===[/]")

    console.print(
        "[yellow]Demonstrating: SoftmaxThreshold (Maximum Softmax Probability)[/]"
    )
    console.print(
        "[dim]Note: DeepGambler and SelectiveNet require training with specialized loss functions,[/]"
    )
    console.print(
        "[dim]so they are not included in this post-hoc evaluation example.[/]"
    )

    # Get test data
    _, test_acc, test_logits, test_labels, _ = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, return_data=True
    )

    # Use SoftmaxThreshold wrapper from the library
    selector = SoftmaxThreshold(backbone=model).to(device)
    selector.eval()

    # Get predictions with confidence using the library's API
    # For already computed logits, we can use them directly
    probs = torch.softmax(test_logits, dim=1)
    confidences = probs.max(dim=1).values
    predictions = test_logits.argmax(dim=1)

    # Sort by confidence (descending)
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_labels = test_labels[sorted_indices]
    sorted_preds = predictions[sorted_indices]

    # Compute risk and coverage at different thresholds
    coverages = []
    risks = []

    for i in range(1, len(sorted_labels) + 1, 50):
        cov = i / len(sorted_labels)
        selected_correct = (sorted_preds[:i] == sorted_labels[:i]).sum().item()
        risk_val = 1 - (selected_correct / i)

        coverages.append(cov)
        risks.append(risk_val)

    # Compute AURC (needs sorted confidence and sorted errors)
    sorted_conf, sorted_idx = confidences.sort(descending=True)
    sorted_errors = (predictions[sorted_idx] != test_labels[sorted_idx]).float()
    aurc_val = aurc(sorted_conf, sorted_errors)

    console.print(f"[yellow]Area Under Risk-Coverage curve (AURC): {aurc_val:.4f}[/]")
    console.print(f"[yellow]Baseline test accuracy: {test_acc:.2f}%[/]")

    # Plot risk-coverage curve
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_risk_coverage(test_logits, test_labels, confidences, ax=ax, show_aurc=True)
    plt.tight_layout()
    plt.savefig(save_dir / "selective_prediction.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]Saved risk-coverage curve to {save_dir / 'selective_prediction.png'}[/]"
    )

    # Show some threshold examples
    table = Table(title="Selective Prediction at Different Thresholds")
    table.add_column("Confidence Threshold", style="cyan")
    table.add_column("Coverage", style="magenta")
    table.add_column("Risk", style="magenta")
    table.add_column("Accuracy", style="magenta")

    for threshold in [0.5, 0.7, 0.9, 0.95, 0.99]:
        # Create reject mask (1 = reject, 0 = accept)
        reject_mask = confidences < threshold
        if (~reject_mask).sum() > 0:  # If there are any accepted samples
            cov = coverage(reject_mask)
            risk_val = risk(predictions, test_labels, reject_mask)
            # Calculate accuracy manually for accepted samples
            accepted_mask = ~reject_mask
            selected_correct = (
                (predictions[accepted_mask] == test_labels[accepted_mask]).sum().item()
            )
            acc = 100.0 * selected_correct / accepted_mask.sum().item()

            table.add_row(
                f"{threshold:.2f}", f"{cov:.4f}", f"{risk_val:.4f}", f"{acc:.2f}%"
            )

    console.print(table)


def demonstrate_conformal_prediction(
    model, cal_loader, test_loader, device, console, save_dir
):
    """Demonstrate conformal prediction methods"""
    console.print("\n[bold cyan]=== CONFORMAL PREDICTION ===[/]")

    # Get calibration and test data
    _, _, cal_logits, cal_labels, _ = evaluate(
        model, cal_loader, nn.CrossEntropyLoss(), device, return_data=True
    )
    _, _, test_logits, test_labels, _ = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, return_data=True
    )

    # Test different alpha values
    alphas = [0.1, 0.05, 0.01]

    results = {}

    for alpha in alphas:
        # Use inductive conformal prediction
        cal_probs = torch.softmax(cal_logits, dim=1)
        test_probs = torch.softmax(test_logits, dim=1)

        # Compute conformal scores (1 - probability of true class)
        cal_scores = 1 - cal_probs[torch.arange(len(cal_labels)), cal_labels]

        # Compute quantile
        n = len(cal_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = torch.quantile(cal_scores, q_level)

        # Create prediction sets (convert boolean mask to list of index tensors)
        pred_sets_bool = test_probs >= (1 - qhat)
        pred_sets = [torch.where(mask)[0] for mask in pred_sets_bool]

        # Compute metrics
        coverage_val = empirical_coverage(test_labels, pred_sets)
        avg_size = average_set_size(pred_sets)

        results[alpha] = {
            "Target Coverage": 1 - alpha,
            "Empirical Coverage": coverage_val,
            "Avg Set Size": avg_size,
        }

    # Display results table
    table = Table(title="Conformal Prediction Results")
    table.add_column("Alpha", style="cyan")
    table.add_column("Target Coverage", style="magenta")
    table.add_column("Empirical Coverage", style="magenta")
    table.add_column("Avg Set Size", style="magenta")

    for alpha, metrics in results.items():
        table.add_row(
            f"{alpha:.2f}",
            f"{metrics['Target Coverage']:.4f}",
            f"{metrics['Empirical Coverage']:.4f}",
            f"{metrics['Avg Set Size']:.2f}",
        )

    console.print(table)

    # Plot coverage vs alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    alphas_plot = np.linspace(0.01, 0.2, 20)
    empirical_coverages = []
    avg_sizes = []

    for alpha in alphas_plot:
        cal_probs = torch.softmax(cal_logits, dim=1)
        test_probs = torch.softmax(test_logits, dim=1)

        cal_scores = 1 - cal_probs[torch.arange(len(cal_labels)), cal_labels]
        n = len(cal_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = torch.quantile(cal_scores, q_level)

        pred_sets_bool = test_probs >= (1 - qhat)
        pred_sets = [torch.where(mask)[0] for mask in pred_sets_bool]

        empirical_coverages.append(empirical_coverage(test_labels, pred_sets))
        avg_sizes.append(average_set_size(pred_sets))

    # Coverage plot
    ax1.plot(
        alphas_plot, [1 - a for a in alphas_plot], "k--", label="Target", linewidth=2
    )
    ax1.plot(alphas_plot, empirical_coverages, "b-", label="Empirical", linewidth=2)
    ax1.set_xlabel("Alpha", fontsize=12)
    ax1.set_ylabel("Coverage", fontsize=12)
    ax1.set_title("Coverage vs Alpha", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set size plot
    ax2.plot(alphas_plot, avg_sizes, "r-", linewidth=2)
    ax2.set_xlabel("Alpha", fontsize=12)
    ax2.set_ylabel("Average Set Size", fontsize=12)
    ax2.set_title("Prediction Set Size vs Alpha", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "conformal_prediction.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]Saved conformal prediction plots to {save_dir / 'conformal_prediction.png'}[/]"
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Comprehensive MNIST example for incerto library"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory for plots"
    )

    # Device selection
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    console = Console()
    console.print("[bold green]Incerto Library - Comprehensive MNIST Example[/]")
    console.print(f"Device: {args.device}")

    # Data loading
    console.print("\n[bold cyan]=== DATA LOADING ===[/]")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # MNIST (in-distribution)
    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Split training into train + calibration
    train_size = int(0.8 * len(trainset))
    cal_size = len(trainset) - train_size
    trainset, calset = random_split(trainset, [train_size, cal_size])

    # FashionMNIST (out-of-distribution)
    ood_testset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    calloader = DataLoader(
        calset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    ood_loader = DataLoader(
        ood_testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    console.print(f"Training samples: {len(trainset)}")
    console.print(f"Calibration samples: {len(calset)}")
    console.print(f"Test samples: {len(testset)}")
    console.print(f"OOD samples: {len(ood_testset)}")

    # Model setup
    console.print("\n[bold cyan]=== MODEL TRAINING ===[/]")
    model = MNISTModel().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    with Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Training", total=args.epochs)

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model, trainloader, optimizer, criterion, args.device
            )
            test_loss, test_acc = evaluate(model, testloader, criterion, args.device)

            console.print(
                f"[bold yellow]Epoch {epoch+1}/{args.epochs}[/] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )
            progress.advance(task)

    # Demonstrate all incerto capabilities
    demonstrate_calibration(
        model, calloader, testloader, args.device, console, save_dir
    )
    demonstrate_ood_detection(
        model, testloader, ood_loader, args.device, console, save_dir
    )
    demonstrate_selective_prediction(model, testloader, args.device, console, save_dir)
    demonstrate_conformal_prediction(
        model, calloader, testloader, args.device, console, save_dir
    )

    console.print(
        "\n[bold green]All demonstrations complete! Check the outputs directory for visualizations.[/]"
    )


if __name__ == "__main__":
    main()
