# Synthetic 2D Examples

Quick conceptual examples using simple 2D synthetic data. Perfect for understanding core uncertainty quantification concepts before applying to real-world problems.

## Why Start Here?

- **Visual**: See concepts in 2D space with clear visualizations
- **Fast**: Each example runs in 15-30 seconds
- **Intuitive**: Build intuition before complexity
- **No dependencies**: Just PyTorch and matplotlib

## Examples

### 1. Calibration Basics
**File**: `calibration_basics.py`
**Runtime**: ~30 seconds
**Difficulty**: ⭐☆☆

Learn what calibration means and why it matters.

**Concepts**:
- What is calibration?
- Overconfident vs well-calibrated models
- Temperature scaling
- Expected Calibration Error (ECE)
- Reliability diagrams

**What you'll see**:
- 2D decision boundaries
- Confidence maps before/after calibration
- Reliability diagrams showing improvement

**Key takeaway**: Models can be accurate but poorly calibrated. Temperature scaling fixes this by adjusting confidence without changing predictions.

---

### 2. OOD Detection Basics
**File**: `ood_detection_basics.py`
**Runtime**: ~20 seconds
**Difficulty**: ⭐☆☆

Learn how to detect out-of-distribution (OOD) data.

**Concepts**:
- What is OOD detection?
- In-distribution vs out-of-distribution
- Maximum Softmax Probability (MSP)
- Energy-based detection
- AUROC metric

**What you'll see**:
- 2D visualization of ID and OOD regions
- Score distributions for different methods
- ROC curves comparing methods

**Key takeaway**: OOD samples often have different score distributions than ID samples. Energy-based methods can outperform simple confidence thresholds.

---

### 3. Conformal Prediction
**File**: `conformal_prediction.py`
**Runtime**: ~15 seconds
**Difficulty**: ⭐☆☆

Learn how to create prediction sets with guaranteed coverage.

**Concepts**:
- What is conformal prediction?
- Inductive conformal prediction
- Coverage guarantees
- Prediction set sizes
- Efficiency vs coverage trade-off

**What you'll see**:
- Prediction sets on 2D data
- Coverage vs miscoverage rate
- Set size distributions

**Key takeaway**: Conformal prediction provides coverage guarantees without assumptions about data distribution. Singleton sets indicate high confidence, larger sets indicate uncertainty.

---

### 4. Selective Prediction
**File**: `selective_prediction.py`
**Runtime**: ~15 seconds
**Difficulty**: ⭐☆☆

Learn when to abstain from making predictions.

**Concepts**:
- What is selective prediction?
- Confidence-based rejection
- Risk-coverage curves
- AURC (Area Under Risk-Coverage)
- Coverage vs accuracy trade-off

**What you'll see**:
- Confidence maps
- Rejection regions
- Risk-coverage curves
- Accuracy improvement from rejection

**Key takeaway**: By rejecting low-confidence predictions, you can significantly improve accuracy on remaining predictions. Useful when cost of error is high.

## Recommended Order

If you're new to uncertainty quantification:

1. **Start with `calibration_basics.py`** - Foundation for understanding confidence
2. **Then `ood_detection_basics.py`** - Learn to detect unusual inputs
3. **Try `selective_prediction.py`** - Learn when to say "I don't know"
4. **Finally `conformal_prediction.py`** - Learn rigorous uncertainty sets

## Running the Examples

```bash
# From the repository root
python examples/01_synthetic/calibration_basics.py
python examples/01_synthetic/ood_detection_basics.py
python examples/01_synthetic/conformal_prediction.py
python examples/01_synthetic/selective_prediction.py
```

All examples save visualizations to `output/synthetic/`.

## What's Next?

After understanding concepts on 2D data, move to:
- **`02_tabular/`** - Apply to real tabular datasets
- **`03_vision/mnist/`** - Scale to image classification
- **`03_vision/cifar/`** - Handle more complex images
- **`04_nlp/`** - Work with language models

## Tips

- **Experiment**: Try changing hyperparameters (smoothing, alpha, thresholds)
- **Read comments**: Code is heavily commented with explanations
- **Check outputs**: All visualizations saved to `output/synthetic/`
- **Modify data**: Try different data distributions in the generation functions

## Common Questions

**Q: Why synthetic data?**
A: Real data has many confounding factors. Synthetic 2D data lets you see exactly what's happening without complexity.

**Q: Can I use these methods on my data?**
A: Yes! These are the same methods used in later examples on real datasets. Start simple, then apply to your problem.

**Q: Which example is most important?**
A: Calibration is fundamental - well-calibrated uncertainty is the foundation for OOD detection, selective prediction, and conformal prediction.

**Q: Do I need to run all examples?**
A: No, but it helps! Each example teaches a different aspect of uncertainty. Together they give a complete picture.

---

**Happy Learning!** Next: [Tabular Examples](../02_tabular/)
