# Incerto Examples

Comprehensive Jupyter notebooks demonstrating uncertainty quantification methods.

## 📓 Interactive Notebooks

**One notebook per incerto module** - comprehensive tutorials from theory to deployment.

| # | Notebook | Module | Description |
|---|----------|--------|-------------|
| 1 | `01_calibration.ipynb` | **calibration** | Confidence calibration (foundational) |
| 2 | `02_ood_detection.ipynb` | **ood** | Out-of-distribution detection |
| 3 | `03_selective_prediction.ipynb` | **sp** | Know when to abstain |
| 4 | `04_conformal_prediction.ipynb` | **conformal** | Prediction sets with guarantees |
| 5 | `05_bayesian_uncertainty.ipynb` | **bayesian** | Bayesian deep learning |
| 6 | `06_active_learning.ipynb` | **active** | Data-efficient learning |
| 7 | `07_shift_detection.ipynb` | **shift** | Distribution shift detection |
| 8 | `08_llm_uncertainty.ipynb` | **llm** | LLM uncertainty |

## 🚀 Quick Start

```bash
# Install incerto with examples dependencies
pip install -e ".[examples]"

# Install Jupyter
pip install jupyter

# Launch notebooks
cd examples
jupyter notebook
```

## 📚 Recommended Order

Follow the numbering **01 → 08** for a logical progression:

1. **Calibration** - Foundation: make model confidence meaningful
2. **OOD Detection** - Detect inputs outside training distribution
3. **Selective Prediction** - Know when to abstain from prediction
4. **Conformal Prediction** - Prediction sets with coverage guarantees
5. **Bayesian Uncertainty** - Principled uncertainty via Bayesian methods
6. **Active Learning** - Use uncertainty for efficient data labeling
7. **Shift Detection** - Monitor for distribution drift in production
8. **LLM Uncertainty** - Specialized methods for language models

## 📖 Notebook Descriptions

### `01_calibration.ipynb`
Learn to make neural network confidence scores meaningful. Modern neural networks are often overconfident - this notebook shows how to fix that using post-hoc calibration methods like temperature scaling, Platt scaling, and isotonic regression. You'll measure calibration with ECE and MCE metrics, visualize with reliability diagrams, and learn to save/load calibrators for production use.

### `02_ood_detection.ipynb`
Detect when your model receives inputs that are unlike anything it saw during training. Out-of-distribution detection is critical for safe deployment. This notebook covers MSP (Maximum Softmax Probability), Energy scores, ODIN, and Mahalanobis distance methods. Evaluate with AUROC and FPR@95TPR, and visualize score distributions to set appropriate thresholds.

### `03_selective_prediction.ipynb`
Sometimes it's better to say "I don't know" than to guess wrong. Selective prediction (also called rejection) lets your model abstain when uncertain. Learn confidence thresholding, risk-coverage trade-offs, and trained rejection methods like SelectiveNet and DeepGambler. Evaluate with AURC and risk-coverage curves.

### `04_conformal_prediction.ipynb`
Get prediction sets with guaranteed coverage - if you want 90% coverage, conformal prediction delivers exactly that. This notebook covers split conformal, cross-validation+ (CV+), and Jackknife+ methods. For classification, learn APS (Adaptive Prediction Sets) and RAPS for smaller, more efficient prediction sets.

### `05_bayesian_uncertainty.ipynb`
Quantify uncertainty the principled way using Bayesian methods. Compare MC Dropout (cheap approximation), Deep Ensembles (gold standard), SWAG (efficient single model), and Laplace Approximation. Understand the difference between epistemic uncertainty (model doesn't know) and aleatoric uncertainty (inherent noise).

### `06_active_learning.ipynb`
Label data efficiently by selecting the most informative samples. Active learning uses uncertainty to prioritize which examples to label next, dramatically reducing annotation costs. This notebook covers uncertainty sampling, BALD, entropy-based selection, diversity sampling, CoreSet, and BADGE strategies with complete training loops.

### `07_shift_detection.ipynb`
Detect when your production data drifts away from training data. Distribution shift is a major cause of ML failures in production. Learn MMD (Maximum Mean Discrepancy), classifier-based detection, KS tests, and energy-based methods. Build monitoring dashboards to catch drift before it impacts users.

### `08_llm_uncertainty.ipynb`
Comprehensive uncertainty quantification for Large Language Models. This notebook covers 45 methods across all categories: token-level (entropy, confidence, perplexity), sequence-level (probability, normalized scores), sampling-based (self-consistency, semantic entropy), verbalized (P(True), self-evaluation), and calibration (temperature scaling, histogram binning). Essential for detecting hallucinations and building trustworthy LLM applications.

## 💡 Tips

- **MPS/CUDA support**: All notebooks auto-detect GPU (including Apple Silicon)
- **Reduce runtime**: Lower epochs or batch size for faster iteration
- **Experiment**: Modify parameters in cells to see effects
- **Production patterns**: Each notebook includes save/load and deployment examples

## 📚 Further Reading

- **Documentation**: [incerto.dev](https://incerto.dev)
- **API Reference**: See docstrings in `incerto/` modules
- **GitHub**: [github.com/steverab/incerto](https://github.com/steverab/incerto)

## ❓ Troubleshooting

**Import errors?**
```bash
pip install -e ".[all]"
```

**Out of memory?**
- Reduce batch size in notebook
- Use CPU: set `device = "cpu"`

**LLM notebook slow?**
- Use smaller model or reduce `n_samples`
- GPU/MPS significantly speeds up inference

---

**Happy Learning! 🎉**
