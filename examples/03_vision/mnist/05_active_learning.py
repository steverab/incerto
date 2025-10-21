"""
Active Learning on MNIST

This example demonstrates active learning strategies for efficient labeling:
- Uncertainty Sampling (Entropy, BALD)
- Diversity Sampling
- Query by Committee

Runtime: ~5-10 minutes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Incerto imports
from incerto.active import (
    EntropyAcquisition,
    BALDAcquisition,
    UncertaintySampling,
    DiversitySampling,
    QueryByCommittee,
)
from incerto.data import get_mnist
from incerto.utils import seed_everything

# Set random seed
seed_everything(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Define Model
# ============================================================================


class SimpleNet(nn.Module):
    """Simple network for MNIST."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# 2. Load Data
# ============================================================================

print("Loading MNIST dataset...")
train_dataset, test_dataset = get_mnist(download=True)

# Use subset for faster demo
pool_size = 2000
train_subset = Subset(train_dataset, range(pool_size))

# Extract all data
X_pool = []
y_pool = []
for x, y in train_subset:
    X_pool.append(x)
    y_pool.append(y)

X_pool = torch.stack(X_pool)
y_pool = torch.tensor(y_pool)

print(f"Pool size: {len(X_pool)}")
print()


# ============================================================================
# 3. Active Learning Function
# ============================================================================


def train_model(X_train, y_train, epochs=5):
    """Train a model on labeled data."""
    model = SimpleNet().to(device)
    model.train()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy."""
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test)
        preds = outputs.argmax(dim=-1)
        accuracy = (preds.cpu() == y_test).float().mean()

    return accuracy.item()


def active_learning_round(
    strategy,
    X_pool,
    y_pool,
    labeled_indices,
    batch_size=100,
):
    """Run one round of active learning."""
    # Get unlabeled indices
    all_indices = set(range(len(X_pool)))
    unlabeled_indices = list(all_indices - set(labeled_indices.tolist()))

    # Train model on labeled data
    X_train = X_pool[labeled_indices]
    y_train = y_pool[labeled_indices]

    model = train_model(X_train, y_train)

    # Query new samples
    X_unlabeled = X_pool[unlabeled_indices]

    if isinstance(strategy, QueryByCommittee):
        # QBC needs multiple models
        selected = strategy.query(X_unlabeled)
    else:
        selected = strategy.query(model, X_unlabeled)

    # Map selected indices back to pool indices
    selected_pool_indices = torch.tensor([unlabeled_indices[i] for i in selected])

    # Update labeled set
    new_labeled = torch.cat([labeled_indices, selected_pool_indices])

    return model, new_labeled


# ============================================================================
# 4. Run Active Learning with Different Strategies
# ============================================================================

# Strategies to compare
strategies = {
    "Entropy": UncertaintySampling(EntropyAcquisition(), batch_size=100),
    "BALD": UncertaintySampling(BALDAcquisition(num_samples=10), batch_size=100),
    "Diversity": DiversitySampling(
        EntropyAcquisition(), batch_size=100, diversity_weight=0.5
    ),
}

# Split test set
test_X = []
test_y = []
for x, y in Subset(test_dataset, range(500)):
    test_X.append(x)
    test_y.append(y)

test_X = torch.stack(test_X)
test_y = torch.tensor(test_y)

# Run active learning
results = {}
num_rounds = 5
initial_labeled = 100

print("Running active learning experiments...\n")

for strategy_name, strategy in strategies.items():
    print(f"Strategy: {strategy_name}")
    print("-" * 40)

    # Initialize with random labeled samples
    labeled_indices = torch.randperm(len(X_pool))[:initial_labeled]

    accuracies = []

    for round_idx in range(num_rounds):
        model, labeled_indices = active_learning_round(
            strategy, X_pool, y_pool, labeled_indices, batch_size=100
        )

        # Evaluate
        acc = evaluate_model(model, test_X, test_y)
        accuracies.append(acc)

        print(f"  Round {round_idx+1}: {len(labeled_indices)} labeled, Acc: {acc:.4f}")

    results[strategy_name] = accuracies
    print()


# ============================================================================
# 5. Visualize Results
# ============================================================================

print("Creating learning curve visualization...")

plt.figure(figsize=(10, 6))

x_axis = [initial_labeled + i * 100 for i in range(num_rounds)]

for strategy_name, accuracies in results.items():
    plt.plot(x_axis, accuracies, marker="o", label=strategy_name, linewidth=2)

plt.xlabel("Number of Labeled Samples", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title(
    "Active Learning: Comparison of Query Strategies", fontsize=14, fontweight="bold"
)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "active_learning.png", dpi=150, bbox_inches="tight")
print(f"Saved visualization to {output_dir / 'active_learning.png'}")

plt.show()


# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "=" * 60)
print("ACTIVE LEARNING SUMMARY")
print("=" * 60)

print(f"{'Strategy':<15} {'Initial Acc':<15} {'Final Acc':<15} {'Improvement':<15}")
print("-" * 60)

for strategy_name, accuracies in results.items():
    initial = accuracies[0]
    final = accuracies[-1]
    improvement = final - initial

    print(f"{strategy_name:<15} {initial:<15.4f} {final:<15.4f} {improvement:<15.4f}")

print("\nDone! Active learning demonstrated.")
print(f"With only {x_axis[-1]} labels, we achieved competitive accuracy!")
