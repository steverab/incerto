"""
OOD Detection Basics on Synthetic 2D Data

Simple demonstration of out-of-distribution (OOD) detection using 2D synthetic data.
Visualizes how different OOD detection methods work.

Concepts covered:
- What is OOD detection?
- Maximum Softmax Probability (MSP)
- Energy-based detection
- ROC curves and AUROC
- Score distributions

Runtime: ~20 seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.ood import MSP, Energy
from incerto.ood.metrics import auroc, fpr_at_tpr
from incerto.ood.visual import plot_roc, score_hist
from incerto.utils import seed_everything

# Set random seed
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Generate Synthetic Data
# ============================================================================


def generate_2d_gaussian_data(n_samples=500, center=[0, 0], std=1.0):
    """Generate 2D Gaussian data."""
    X = np.random.randn(n_samples, 2) * std + np.array(center)
    return X.astype(np.float32)


print("Generating synthetic data...")

# In-distribution: Two clusters
X_id_1 = generate_2d_gaussian_data(250, center=[-2, 0], std=0.5)
X_id_2 = generate_2d_gaussian_data(250, center=[2, 0], std=0.5)
X_id = np.vstack([X_id_1, X_id_2])
y_id = np.concatenate([np.zeros(250), np.ones(250)]).astype(np.int64)

# Out-of-distribution: Different regions
X_ood_1 = generate_2d_gaussian_data(200, center=[0, 3], std=0.4)
X_ood_2 = generate_2d_gaussian_data(200, center=[0, -3], std=0.4)
X_ood = np.vstack([X_ood_1, X_ood_2])

print(f"ID samples: {len(X_id)}")
print(f"OOD samples: {len(X_ood)}\n")


# ============================================================================
# 2. Train Simple Classifier
# ============================================================================


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


print("Training classifier on ID data...")
model = SimpleMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

X_id_t = torch.FloatTensor(X_id).to(device)
y_id_t = torch.LongTensor(y_id).to(device)
X_ood_t = torch.FloatTensor(X_ood).to(device)

# Train
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_id_t)
    loss = criterion(outputs, y_id_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 25 == 0:
        acc = (outputs.argmax(1) == y_id_t).float().mean() * 100
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2f}%")

print()


# ============================================================================
# 3. OOD Detection Methods
# ============================================================================

print("Evaluating OOD detection methods...\n")

# Method 1: Maximum Softmax Probability (MSP)
msp_detector = MSP(model)
id_scores_msp = msp_detector.score(X_id_t).cpu()
ood_scores_msp = msp_detector.score(X_ood_t).cpu()

auroc_msp = auroc(id_scores_msp, ood_scores_msp)
fpr95_msp = fpr_at_tpr(id_scores_msp, ood_scores_msp, tpr=0.95)

print("Maximum Softmax Probability (MSP):")
print(f"  AUROC: {auroc_msp:.4f}")
print(f"  FPR@95: {fpr95_msp:.4f}")

# Method 2: Energy Score
energy_detector = Energy(model, temperature=1.0)
id_scores_energy = energy_detector.score(X_id_t).cpu()
ood_scores_energy = energy_detector.score(X_ood_t).cpu()

auroc_energy = auroc(id_scores_energy, ood_scores_energy)
fpr95_energy = fpr_at_tpr(id_scores_energy, ood_scores_energy, tpr=0.95)

print("\nEnergy Score:")
print(f"  AUROC: {auroc_energy:.4f}")
print(f"  FPR@95: {fpr95_energy:.4f}")
print()


# ============================================================================
# 4. Visualizations
# ============================================================================

output_dir = Path("output/synthetic")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Data distribution with decision boundary
fig, ax = plt.subplots(figsize=(10, 8))

# Create mesh for decision boundary
h = 0.05
x_min, x_max = -4, 4
y_min, y_max = -4, 4
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

model.eval()
with torch.no_grad():
    Z = model(grid).cpu().numpy()
    Z_probs = F.softmax(torch.FloatTensor(Z), dim=1).numpy()
    Z_conf = Z_probs.max(axis=1)

Z_conf = Z_conf.reshape(xx.shape)

# Plot decision boundary and confidence
contour = ax.contourf(xx, yy, Z_conf, levels=20, cmap="RdYlGn", alpha=0.6)
plt.colorbar(contour, ax=ax, label="Confidence")

# Plot data points
ax.scatter(
    X_id[:, 0],
    X_id[:, 1],
    c=y_id,
    cmap="RdYlBu",
    edgecolors="k",
    s=50,
    label="ID data",
    marker="o",
)
ax.scatter(
    X_ood[:, 0],
    X_ood[:, 1],
    c="purple",
    edgecolors="k",
    s=50,
    label="OOD data",
    marker="X",
)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Data Distribution and Model Confidence")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(output_dir / "data_distribution.png", dpi=150, bbox_inches="tight")
print(f"Saved data distribution to {output_dir / 'data_distribution.png'}")

# Plot 2: ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_roc(id_scores_msp, ood_scores_msp, ax=axes[0], label="MSP")
axes[0].set_title(f"MSP (AUROC={auroc_msp:.4f})")

plot_roc(id_scores_energy, ood_scores_energy, ax=axes[1], label="Energy")
axes[1].set_title(f"Energy (AUROC={auroc_energy:.4f})")

plt.tight_layout()
plt.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved ROC curves to {output_dir / 'roc_curves.png'}")

# Plot 3: Score distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

score_hist(id_scores_msp, ood_scores_msp, ax=axes[0])
axes[0].set_title("MSP Score Distribution")
axes[0].set_xlabel("Softmax Probability")

score_hist(id_scores_energy, ood_scores_energy, ax=axes[1])
axes[1].set_title("Energy Score Distribution")
axes[1].set_xlabel("Energy")

plt.tight_layout()
plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
print(f"Saved score distributions to {output_dir / 'score_distributions.png'}")


# ============================================================================
# 5. Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"MSP:    AUROC={auroc_msp:.4f}, FPR@95={fpr95_msp:.4f}")
print(f"Energy: AUROC={auroc_energy:.4f}, FPR@95={fpr95_energy:.4f}")
print()
print("Key Takeaways:")
print("• OOD samples typically have lower confidence (MSP)")
print("• Energy score often more robust than MSP")
print("• Good OOD detection: high AUROC, low FPR@95")
print("• Score distributions should be well-separated")
print("=" * 60)
