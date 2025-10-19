# Incerto Examples

Comprehensive examples demonstrating uncertainty quantification methods across different data modalities.

## 📁 Directory Structure

```
examples/
├── 01_synthetic/       # Quick conceptual examples (~1 minute each)
├── 02_tabular/         # Tabular data examples (~1-5 minutes each)
├── 03_vision/          # Computer vision examples (~5-30 minutes each)
│   ├── mnist/          # MNIST examples
│   └── cifar/          # CIFAR-10 examples
└── 04_nlp/             # Natural language processing examples
```

## 🎯 Learning Paths

### Path 1: Beginner (Concepts First)
**Goal:** Understand core uncertainty concepts with simple visualizations

1. **`01_synthetic/calibration_basics.py`** - What is calibration?
2. **`01_synthetic/ood_detection_basics.py`** - What is OOD detection?
3. **`01_synthetic/conformal_prediction.py`** - Prediction sets with guarantees
4. **`01_synthetic/selective_prediction.py`** - When to abstain from prediction
5. **`02_tabular/posthoc_calibration.py`** - Apply to real tabular data

**Time:** ~1 hour | **Difficulty:** ⭐☆☆

### Path 2: Practitioner (Real Applications)
**Goal:** Apply methods to practical problems

1. **`02_tabular/posthoc_calibration.py`** - Tabular classification
2. **`03_vision/mnist/01_posthoc_evaluation.py`** - Image classification
3. **`02_tabular/training_methods.py`** - Training-time improvements
4. **`03_vision/mnist/02_training_methods.py`** - Training methods on images
5. **`03_vision/mnist/03_ensemble_methods.py`** - Ensemble approaches

**Time:** ~2-3 hours | **Difficulty:** ⭐⭐☆

### Path 3: Researcher (Advanced Methods)
**Goal:** Master cutting-edge techniques

1. **`03_vision/cifar/01_posthoc_evaluation.py`** - Complex images
2. **`03_vision/cifar/02_training_methods.py`** - Training methods
3. **`03_vision/cifar/03_advanced_training.py`** - Evidential, focal loss
4. **`04_nlp/llm_uncertainty.py`** - LLM uncertainty quantification

**Time:** ~4-6 hours | **Difficulty:** ⭐⭐⭐

## 📊 Examples by Topic

### Calibration
- ✅ `01_synthetic/calibration_basics.py` - Visualization on 2D data
- ✅ `02_tabular/posthoc_calibration.py` - Post-hoc methods
- ✅ `02_tabular/training_methods.py` - Label smoothing, focal loss
- ✅ `03_vision/mnist/01_posthoc_evaluation.py` - Temperature scaling
- ✅ `03_vision/cifar/03_advanced_training.py` - Advanced methods

### OOD Detection
- ✅ `01_synthetic/ood_detection_basics.py` - Visualization on 2D data
- ✅ `03_vision/mnist/01_posthoc_evaluation.py` - MSP, Energy, ODIN
- ✅ `03_vision/mnist/02_training_methods.py` - Mixup for OOD
- ✅ `03_vision/cifar/01_posthoc_evaluation.py` - CIFAR vs SVHN

### Selective Prediction
- ✅ `01_synthetic/selective_prediction.py` - Risk-coverage curves
- ✅ `03_vision/mnist/01_posthoc_evaluation.py` - Confidence thresholding
- ✅ `03_vision/mnist/02_training_methods.py` - Self-Adaptive Training

### Conformal Prediction
- ✅ `01_synthetic/conformal_prediction.py` - Prediction sets
- ✅ `03_vision/mnist/01_posthoc_evaluation.py` - Inductive CP

### Ensemble Methods
- ✅ `03_vision/mnist/03_ensemble_methods.py` - Deep ensembles vs MC Dropout

### LLM Uncertainty
- ✅ `04_nlp/llm_uncertainty.py` - Token, sequence, sampling-based methods

## 🚀 Quick Start

### Run Your First Example

```bash
# Simple 2D visualization (30 seconds)
python examples/01_synthetic/calibration_basics.py

# Real dataset (1 minute)
python examples/02_tabular/posthoc_calibration.py

# Image classification (5 minutes)
python examples/03_vision/mnist/01_posthoc_evaluation.py
```

### Requirements

```bash
# Install incerto
pip install -e .

# Or with dependencies
pip install -e ".[examples]"
```

## 📖 Example Descriptions

### 01_synthetic/ - Toy Examples
Perfect for understanding concepts before applying to real data.

| File | Description | Runtime | Key Concepts |
|------|-------------|---------|--------------|
| `calibration_basics.py` | 2D calibration visualization | ~30s | Temperature scaling, ECE, reliability diagrams |
| `ood_detection_basics.py` | 2D OOD detection | ~20s | MSP, Energy scores, AUROC, score distributions |
| `conformal_prediction.py` | Prediction sets demo | ~15s | Coverage guarantees, set sizes, efficiency |
| `selective_prediction.py` | Risk-coverage trade-offs | ~15s | Rejection, AURC, coverage vs accuracy |

### 02_tabular/ - Tabular Data
Real-world tabular classification with uncertainty.

| File | Description | Runtime | Dataset |
|------|-------------|---------|---------|
| `posthoc_calibration.py` | Post-hoc calibration methods | ~10s | Wine (178 samples) |
| `training_methods.py` | Training-time methods | ~30s | Breast Cancer (569 samples) |

### 03_vision/mnist/ - MNIST Examples
Handwritten digit classification (28x28 grayscale).

| File | Description | Runtime | Methods |
|------|-------------|---------|---------|
| `01_posthoc_evaluation.py` | Comprehensive post-hoc evaluation | ~5 min | Calibration, OOD, Selective, Conformal |
| `02_training_methods.py` | Training method comparison | ~5 min | Baseline, Label Smoothing, Focal, Mixup, SAT |
| `03_ensemble_methods.py` | Ensemble approaches | ~10 min | Single model, Deep Ensemble, MC Dropout |

### 03_vision/cifar/ - CIFAR-10 Examples
Natural images (32x32 RGB, 10 classes).

| File | Description | Runtime | Architecture |
|------|-------------|---------|--------------|
| `01_posthoc_evaluation.py` | Post-hoc methods | ~10 min | ResNet-18 |
| `02_training_methods.py` | Training methods | ~30 min | ResNet-18 |
| `03_advanced_training.py` | Advanced techniques | ~30 min | Evidential, Focal, Confidence Penalty |

### 04_nlp/ - NLP Examples
Natural language processing with uncertainty.

| File | Description | Runtime | Model |
|------|-------------|---------|-------|
| `llm_uncertainty.py` | LLM uncertainty quantification | ~5 min | Qwen2.5-0.5B or similar |

## 💡 Tips

### For Learning
1. **Start with synthetic examples** - Visualize concepts before complexity
2. **Follow a learning path** - Structured progression
3. **Read the code comments** - Detailed explanations throughout
4. **Check output/** - All examples save visualizations

### For Development
1. **Use examples as templates** - Copy and modify for your use case
2. **Check incerto.utils** - Reusable components (models, training loops)
3. **Mix and match** - Combine methods from different examples
4. **Contribute** - Add your own examples!

### For Debugging
1. **Reduce data size** - Use smaller subsets for faster iteration
2. **Reduce epochs** - 2-3 epochs enough to test code
3. **Use CPU** - Set `device = 'cpu'` if GPU unavailable
4. **Check outputs** - Examples save results to `output/` directory

## 🎨 Visualizations

All examples save visualizations to the `output/` directory:

```
output/
├── synthetic/          # 2D plots, decision boundaries
├── tabular/            # Comparison tables, bar charts
├── vision/
│   ├── mnist/          # Reliability diagrams, ROC curves
│   └── cifar/          # Training curves, distributions
└── nlp/                # Token-level, sequence-level plots
```

## 📚 Further Reading

- **Documentation:** [docs.incerto.ai](https://docs.incerto.ai) (TODO)
- **API Reference:** See docstrings in `incerto/` modules
- **Papers:** See references in individual examples
- **Tutorials:** See `tutorials/` directory

## 🤝 Contributing

Have a great example? Contribute!

1. Follow the existing structure
2. Include docstring with description, runtime, concepts
3. Save visualizations to `output/`
4. Add entry to this README
5. Submit PR

## ❓ Troubleshooting

**Out of memory?**
- Reduce batch size
- Use smaller model
- Reduce dataset size

**Too slow?**
- Use GPU if available
- Reduce epochs
- Use smaller dataset

**Import errors?**
- Install incerto: `pip install -e .`
- Check dependencies: `pip install -r requirements.txt`

**Other issues?**
- Check GitHub issues
- Ask on Discord
- Open new issue

---

**Happy Learning! 🎉**
