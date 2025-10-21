"""
Bayesian Deep Learning Methods on MNIST

This example demonstrates Bayesian approaches for uncertainty quantification:
- MC Dropout
- Deep Ensembles
- SWAG

Runtime: ~5-10 minutes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.bayesian import MCDropout, DeepEnsemble, SWAG
from incerto.bayesian.utils import decompose_uncertainty
from incerto.data import get_mnist
from incerto.utils import seed_everything

# Set random seed
seed_everything(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Define Model Architecture
# ============================================================================


class ConvNet(nn.Module):
    """Simple ConvNet for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# ============================================================================
# 2. Load Data
# ============================================================================

print("Loading MNIST dataset...")
train_dataset, test_dataset = get_mnist(download=True)

# Use subset for faster demo
train_subset = Subset(train_dataset, range(5000))
test_subset = Subset(test_dataset, range(1000))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)

print(f"Train samples: {len(train_subset)}")
print(f"Test samples: {len(test_subset)}\n")


# ============================================================================
# 3. Train a Base Model
# ============================================================================


def train_model(model, train_loader, epochs=3):
    """Train a model."""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model


print("Training base model...")
base_model = train_model(ConvNet(), train_loader)
print()


# ============================================================================
# 4. MC Dropout
# ============================================================================

print("=" * 60)
print("MC DROPOUT")
print("=" * 60)

mc_model = MCDropout(base_model, num_samples=20)

# Get predictions on a batch
test_x, test_y = next(iter(test_loader))
test_x = test_x.to(device)

mean_preds, variance = mc_model.predict(test_x)
entropy = mc_model.predict_entropy(test_x)
mi = mc_model.predict_mutual_information(test_x)

print(f"Mean prediction shape: {mean_preds.shape}")
print(f"Variance shape: {variance.shape}")
print(f"Average predictive entropy: {entropy.mean():.4f}")
print(f"Average mutual information: {mi.mean():.4f}")
print()


# ============================================================================
# 5. Deep Ensembles
# ============================================================================

print("=" * 60)
print("DEEP ENSEMBLES")
print("=" * 60)


def create_model():
    return ConvNet()


ensemble = DeepEnsemble(create_model, num_models=5)

print("Training ensemble members...")
for i in range(5):
    print(f"\nTraining model {i+1}/5...")
    optimizer = torch.optim.Adam(ensemble.models[i].parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    ensemble.train_member(
        i, train_loader, optimizer, criterion, num_epochs=3, device=device
    )

# Get ensemble predictions
mean_preds_ens, variance_ens = ensemble.predict(test_x)
diversity = ensemble.diversity(test_x)

print(f"\nEnsemble mean prediction shape: {mean_preds_ens.shape}")
print(f"Ensemble variance shape: {variance_ens.shape}")
print(f"Average ensemble diversity: {diversity.mean():.4f}")
print()


# ============================================================================
# 6. SWAG
# ============================================================================

print("=" * 60)
print("SWAG (Stochastic Weight Averaging - Gaussian)")
print("=" * 60)

swag_model = SWAG(ConvNet(), num_samples=20, max_models=10)

# Train and collect SWAG statistics
print("Training and collecting SWAG statistics...")
base_swag = ConvNet()
base_swag = train_model(base_swag, train_loader, epochs=2)

# Collect models (in practice, do this during training)
for _ in range(10):
    swag_model.collect_model(base_swag)

# Get SWAG predictions
mean_preds_swag, variance_swag = swag_model.predict(test_x)

print(f"SWAG mean prediction shape: {mean_preds_swag.shape}")
print(f"SWAG variance shape: {variance_swag.shape}")
print()


# ============================================================================
# 7. Uncertainty Decomposition
# ============================================================================

print("=" * 60)
print("UNCERTAINTY DECOMPOSITION")
print("=" * 60)

# Get MC samples for decomposition
_, _, samples = mc_model.predict(test_x, return_samples=True)

total_unc, epistemic_unc, aleatoric_unc = decompose_uncertainty(samples)

print(f"Total uncertainty: {total_unc.mean():.4f}")
print(f"Epistemic uncertainty: {epistemic_unc.mean():.4f}")
print(f"Aleatoric uncertainty: {aleatoric_unc.mean():.4f}")
print()


# ============================================================================
# 8. Visualize Uncertainty
# ============================================================================

print("Creating uncertainty visualizations...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Show 10 test images with their uncertainties
for idx in range(10):
    ax = axes[idx // 5, idx % 5]
    ax.imshow(test_x[idx].cpu().squeeze(), cmap="gray")

    pred_class = mean_preds[idx].argmax().item()
    true_class = test_y[idx].item()
    unc = entropy[idx].item()

    color = "green" if pred_class == true_class else "red"
    ax.set_title(
        f"Pred: {pred_class}\nTrue: {true_class}\nUnc: {unc:.2f}",
        color=color,
        fontsize=8,
    )
    ax.axis("off")

plt.tight_layout()

# Save figure
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "bayesian_uncertainty.png", dpi=150, bbox_inches="tight")
print(f"Saved visualization to {output_dir / 'bayesian_uncertainty.png'}")

plt.show()


# ============================================================================
# 9. Compare Methods
# ============================================================================

print("\n" + "=" * 60)
print("METHOD COMPARISON")
print("=" * 60)

print(f"{'Method':<20} {'Avg Uncertainty':<20} {'Avg Variance':<20}")
print("-" * 60)
print(f"{'MC Dropout':<20} {entropy.mean():.4f} {'':>15} {variance.mean():.4f}")
print(f"{'Deep Ensemble':<20} {'-':<20} {variance_ens.mean():.4f}")
print(f"{'SWAG':<20} {'-':<20} {variance_swag.mean():.4f}")
print()

print("\nDone! All Bayesian methods demonstrated.")
