"""
Advanced Training Tutorial: Uncertainty-Aware Training Methods

This tutorial demonstrates advanced training-time techniques for improving
uncertainty quantification in deep neural networks.

Topics covered:
1. Evidential Deep Learning - Predicting uncertainty directly
2. Focal Loss - Handling hard examples and improving calibration
3. Deep Ensembles with diversity - Encouraging diverse predictions
4. Temperature-Aware Training - Learning temperature during training
5. Confidence Penalty - Preventing overconfidence

Each method includes:
- Theoretical motivation
- Implementation details
- Training curves
- Comparative evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from rich.console import Console

# Incerto imports
from incerto.data import CIFAR10, CIFAR10_vs_SVHN, create_dataloaders
from incerto.calibration import ece_score
from incerto.ood import auroc as ood_auroc
from incerto.sp import aurc

console = Console()


# ============================================================================
# Section 1: Evidential Deep Learning
# ============================================================================


class EvidentialNet(nn.Module):
    """
    Evidential Neural Network for CIFAR-10.

    Instead of predicting class probabilities, predicts Dirichlet parameters
    that represent second-order uncertainty (uncertainty about uncertainty).

    Reference: Sensoy et al. "Evidential Deep Learning to Quantify
    Classification Uncertainty" (NeurIPS 2018)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Evidential output (predicts evidence for each class)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass returns evidence (non-negative values).
        Evidence can be converted to Dirichlet parameters: alpha = evidence + 1
        """
        features = self.features(x)
        evidence = self.fc(features)
        # Ensure evidence is non-negative
        evidence = F.softplus(evidence)
        return evidence

    def get_uncertainty(self, evidence):
        """
        Compute various uncertainty measures from evidence.

        Returns:
            - alpha: Dirichlet parameters
            - belief: Predicted probabilities
            - uncertainty: Total uncertainty (vacuity)
            - epistemic: Epistemic uncertainty
            - aleatoric: Aleatoric uncertainty
        """
        alpha = evidence + 1
        S = alpha.sum(dim=1, keepdim=True)

        belief = alpha / S
        uncertainty = self.num_classes / S

        # Epistemic uncertainty (model uncertainty)
        expected_prob = alpha / S
        epistemic = (alpha * (S - alpha)) / (S * S * (S + 1))
        epistemic = epistemic.sum(dim=1, keepdim=True)

        return {
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty,
            "epistemic": epistemic,
        }


def evidential_loss(outputs, targets, epoch, num_epochs, device):
    """
    Evidential loss with KL divergence annealing.

    Loss = MSE loss + lambda * KL(Dir(alpha) || Dir(1))

    The KL term acts as a regularizer, preventing the model from being
    overconfident on incorrect predictions.
    """
    alpha = outputs + 1
    S = alpha.sum(dim=1, keepdim=True)

    # One-hot encode targets
    targets_one_hot = F.one_hot(targets, num_classes=alpha.size(1)).float()

    # MSE loss between predicted probabilities and true labels
    prob = alpha / S
    mse_loss = ((targets_one_hot - prob) ** 2).sum(dim=1).mean()

    # KL divergence regularization
    # Encourages the model to have high uncertainty on wrong predictions
    alpha_tilde = targets_one_hot + (1 - targets_one_hot) * alpha

    # KL[Dir(alpha_tilde) || Dir(1)]
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    first_term = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(
        dim=1, keepdim=True
    )
    second_term = (
        (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
    ).sum(dim=1, keepdim=True)
    kl_loss = (first_term + second_term).mean()

    # Anneal KL coefficient from 0 to 1
    kl_coeff = min(1.0, epoch / (num_epochs * 0.5))

    total_loss = mse_loss + kl_coeff * kl_loss

    return total_loss, mse_loss, kl_loss


# ============================================================================
# Section 2: Focal Loss
# ============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples and class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The (1 - p_t)^gamma term down-weights easy examples, allowing the model
    to focus on hard examples. This improves calibration by preventing
    overconfidence on easy examples.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (N, C)
            targets: Ground truth labels (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ResNet18(nn.Module):
    """Simplified ResNet18 for CIFAR-10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block may have stride > 1
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ============================================================================
# Section 3: Confidence Penalty
# ============================================================================


class ConfidencePenalty(nn.Module):
    """
    Confidence penalty to prevent overconfidence.

    Adds a penalty term that encourages the model to have higher entropy
    (lower confidence), preventing overconfident predictions.

    Loss = CE + beta * Confidence_Penalty

    where Confidence_Penalty = -H(p) (negative entropy)
    """

    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs (N, C)
            targets: Ground truth labels (N,)
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets)

        # Confidence penalty (negative entropy)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        confidence_penalty = -entropy  # Negative entropy = confidence

        total_loss = ce_loss + self.beta * confidence_penalty

        return total_loss


# ============================================================================
# Section 4: Temperature-Aware Training
# ============================================================================


class TemperatureNet(nn.Module):
    """
    Neural network with learnable temperature parameter.

    Instead of post-hoc temperature scaling, learns the temperature
    during training for better calibration.
    """

    def __init__(self, backbone, init_temp=1.0):
        super().__init__()
        self.backbone = backbone
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, x, return_unscaled=False):
        logits = self.backbone(x)

        if return_unscaled:
            return logits

        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        return scaled_logits


# ============================================================================
# Training Functions
# ============================================================================


def train_evidential(model, train_loader, val_loader, device, epochs=50):
    """Train evidential neural network."""
    console.print("[cyan]Training Evidential Deep Learning model...[/cyan]")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_kl = 0.0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            evidence = model(inputs)
            loss, mse_loss, kl_loss = evidential_loss(
                evidence, targets, epoch, epochs, device
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_kl += kl_loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                evidence = model(inputs)
                alpha = evidence + 1
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                _, predicted = probs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            console.print(
                f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f} "
                f"(MSE={epoch_mse/len(train_loader):.4f}, KL={epoch_kl/len(train_loader):.4f}), "
                f"Val Acc={val_acc:.2f}%"
            )

    return model, train_losses, val_accs


def train_with_focal_loss(
    model, train_loader, val_loader, device, epochs=50, gamma=2.0
):
    """Train with focal loss."""
    console.print(f"[cyan]Training with Focal Loss (γ={gamma})...[/cyan]")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = FocalLoss(gamma=gamma)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            console.print(
                f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%"
            )

    return model, train_losses, val_accs


def train_with_confidence_penalty(
    model, train_loader, val_loader, device, epochs=50, beta=0.1
):
    """Train with confidence penalty."""
    console.print(f"[cyan]Training with Confidence Penalty (β={beta})...[/cyan]")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = ConfidencePenalty(beta=beta)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * correct / total
        train_losses.append(epoch_loss / len(train_loader))
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            console.print(
                f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%"
            )

    return model, train_losses, val_accs


def main():
    """Run advanced training tutorial."""
    console.print(
        "[bold green]Advanced Training Tutorial: Uncertainty-Aware Methods[/bold green]\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}\n")

    # Load CIFAR-10
    console.print("[yellow]Loading CIFAR-10...[/yellow]")
    cifar = CIFAR10(root="./data", val_split=0.1, normalize=True, augmentation=True)
    train_dataset, val_dataset, test_dataset = cifar.get_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=128,
        num_workers=2,
    )

    console.print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n"
    )

    # Train models
    epochs = 20  # Reduced for demo; use 50+ for real experiments

    console.print("\n[bold]═══ Training Models ═══[/bold]\n")

    # 1. Evidential Deep Learning
    console.print("[bold cyan]1. Evidential Deep Learning[/bold cyan]")
    edl_model = EvidentialNet(num_classes=10)
    edl_model, edl_losses, edl_accs = train_evidential(
        edl_model, train_loader, val_loader, device, epochs
    )

    # 2. Focal Loss
    console.print("\n[bold cyan]2. Focal Loss[/bold cyan]")
    focal_model = ResNet18(num_classes=10)
    focal_model, focal_losses, focal_accs = train_with_focal_loss(
        focal_model, train_loader, val_loader, device, epochs, gamma=2.0
    )

    # 3. Confidence Penalty
    console.print("\n[bold cyan]3. Confidence Penalty[/bold cyan]")
    cp_model = ResNet18(num_classes=10)
    cp_model, cp_losses, cp_accs = train_with_confidence_penalty(
        cp_model, train_loader, val_loader, device, epochs, beta=0.1
    )

    console.print("\n[bold green]✅ Training complete![/bold green]")

    # Plot training curves
    console.print("\n[yellow]Plotting training curves...[/yellow]")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(edl_losses, label="Evidential")
    ax1.plot(focal_losses, label="Focal Loss")
    ax1.plot(cp_losses, label="Confidence Penalty")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(edl_accs, label="Evidential")
    ax2.plot(focal_accs, label="Focal Loss")
    ax2.plot(cp_accs, label="Confidence Penalty")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]Saved training curves to {output_dir / 'training_curves.png'}[/green]"
    )

    console.print("\n[bold green]Tutorial complete![/bold green]")
    console.print("\n[yellow]Key insights:[/yellow]")
    console.print(
        "• Evidential: Directly models uncertainty as second-order distributions"
    )
    console.print("• Focal Loss: Focuses on hard examples, improves calibration")
    console.print(
        "• Confidence Penalty: Prevents overconfidence via entropy regularization"
    )


if __name__ == "__main__":
    main()
