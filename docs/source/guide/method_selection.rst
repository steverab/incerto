Which Method Should I Use?
==========================

This guide helps you choose the right uncertainty quantification method for your use case.
Start with your **goal**, then narrow down based on your constraints.

Step 1: What's Your Goal?
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Goal
     - Description
     - Module
   * - **Trustworthy confidence scores**
     - Model says 80% confident → should be right 80% of the time
     - :doc:`Calibration </guide/calibration>`
   * - **Detect anomalous inputs**
     - Flag inputs that are unlike training data
     - :doc:`OOD Detection </guide/ood>`
   * - **Prediction sets with guarantees**
     - Return a set of possible labels with provable coverage
     - :doc:`Conformal Prediction </guide/conformal>`
   * - **Know when to abstain**
     - Reject uncertain predictions, send to human review
     - :doc:`Selective Prediction </guide/selective>`
   * - **Rich uncertainty estimates**
     - Separate "I don't know" (epistemic) from "it's inherently noisy" (aleatoric)
     - :doc:`Bayesian Methods </guide/bayesian>`
   * - **Detect distribution shift**
     - Monitor if production data drifts from training data
     - :doc:`Shift Detection </guide/shift>`
   * - **LLM reliability**
     - Assess when a language model is likely hallucinating
     - :doc:`LLM Uncertainty </guide/llm>`
   * - **Efficient labeling**
     - Choose which unlabeled samples to annotate next
     - :doc:`Active Learning </guide/active>`


Step 2: Choosing a Specific Method
-----------------------------------

Calibration
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Method
     - When to Use
     - Data Needed
     - Notes
   * - **TemperatureScaling**
     - Default first choice
     - Validation set (logits + labels)
     - Single parameter, fast, works well for most models
   * - **VectorScaling**
     - Per-class calibration needed
     - Validation set
     - One temperature per class
   * - **MatrixScaling**
     - Maximum flexibility
     - Large validation set
     - Full affine transform; risk of overfitting with small val sets
   * - **PlattScalingCalibrator**
     - Binary classification
     - Validation set
     - Classic method via logistic regression
   * - **IsotonicRegressionCalibrator**
     - Non-parametric fit
     - Larger validation set (100+)
     - More expressive than Temperature Scaling but needs more data
   * - **DirichletCalibrator**
     - Multiclass, flexible
     - Validation set
     - Generalizes Temperature, Vector, and Matrix scaling
   * - **BetaCalibrator**
     - Binary or multiclass
     - Validation set
     - Beta distribution mapping; falls back to isotonic for multiclass
   * - **LabelSmoothingLoss**
     - During training
     - Training data
     - Simple regularizer, improves calibration as a side effect
   * - **FocalLoss**
     - Class imbalance + calibration
     - Training data
     - Down-weights easy examples; good for imbalanced datasets

**Rule of thumb**: Start with ``TemperatureScaling``. If ECE is still too high, try ``IsotonicRegressionCalibrator`` (if you have enough validation data) or ``DirichletCalibrator``.

OOD Detection
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Method
     - When to Use
     - Requirements
     - Notes
   * - **MSP**
     - Simplest baseline
     - Trained model only
     - 1 - max softmax probability; no tuning needed
   * - **Energy**
     - Better than MSP, still simple
     - Trained model only
     - Log-sum-exp of logits; stronger signal than MSP
   * - **MaxLogit**
     - Large output spaces
     - Trained model only
     - Negative max logit; good for many-class problems
   * - **ODIN**
     - Improved separation
     - Trained model only
     - Temperature scaling + input perturbation; needs tuning T and epsilon
   * - **Mahalanobis**
     - Feature-space detection
     - In-distribution data to fit
     - Uses class-conditional Gaussians in feature space; strong performance
   * - **KNN**
     - Non-parametric feature-space
     - In-distribution data to fit
     - k-nearest neighbor distances; no distributional assumptions

**Rule of thumb**: Start with ``Energy`` (best effort-to-performance ratio). For stronger detection when you have in-distribution data available, use ``Mahalanobis`` or ``KNN``.

Conformal Prediction
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Method
     - When to Use
     - Task
     - Notes
   * - **inductive_conformal**
     - Standard conformal
     - Classification
     - Simple threshold on softmax; valid coverage guarantee
   * - **aps**
     - Adaptive sets
     - Classification
     - Adaptive Prediction Sets; smaller sets for easy examples
   * - **raps**
     - Tighter sets
     - Classification
     - Regularized APS; penalizes large sets for cleaner results
   * - **mondrian_conformal**
     - Class-conditional
     - Classification
     - Separate thresholds per class; useful for imbalanced data
   * - **jackknife_plus**
     - Prediction intervals
     - Regression
     - Leave-one-out based; needs N model fits
   * - **cv_plus**
     - Efficient intervals
     - Regression
     - K-fold based; more practical than jackknife+

**Rule of thumb**: For classification, use ``raps`` (tightest sets with guarantees). For regression, use ``cv_plus`` (good tradeoff between efficiency and coverage).

Selective Prediction
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Method
     - When to Use
     - Requirements
     - Notes
   * - **SoftmaxThreshold**
     - Simple confidence gating
     - Trained model only
     - Threshold on softmax; no additional training
   * - **SelfAdaptiveTraining**
     - Noisy labels
     - Training time
     - Learns to correct its own labels during training
   * - **DeepGambler**
     - Learned abstention
     - Training time
     - Adds an abstention class; trains jointly with task
   * - **SelectiveNet**
     - End-to-end selective
     - Training time
     - Separate selection head; jointly optimizes coverage and risk

**Rule of thumb**: ``SoftmaxThreshold`` for post-hoc (no retraining). ``SelectiveNet`` for best selective accuracy when you can retrain.

Bayesian Methods
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Method
     - When to Use
     - Cost
     - Notes
   * - **MCDropout**
     - Quick epistemic uncertainty
     - Low (existing model + dropout)
     - Enable dropout at test time; cheap but approximate
   * - **DeepEnsemble**
     - Gold standard
     - High (train N models)
     - Best calibrated uncertainty; trains multiple independent models
   * - **SWAG**
     - Single-model uncertainty
     - Medium (collect SGD iterates)
     - Gaussian over weights from SGD trajectory
   * - **LaplaceApproximation**
     - Post-hoc Bayesian
     - Low (post-training)
     - Gaussian around MAP; no retraining needed
   * - **VariationalBayesNN**
     - Full Bayesian
     - High (variational training)
     - Bayes by Backprop; most principled but hardest to train

**Rule of thumb**: ``MCDropout`` if your model already has dropout (easiest). ``DeepEnsemble`` for best results when compute allows. ``LaplaceApproximation`` for post-hoc Bayesian on an existing model.

Shift Detection
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Method
     - When to Use
     - Requirements
     - Notes
   * - **MMDShiftDetector**
     - General covariate shift
     - Reference data
     - Maximum Mean Discrepancy with kernel; powerful general test
   * - **KSShiftDetector**
     - Per-feature shift
     - Reference data
     - Kolmogorov-Smirnov test; good for low-dimensional data
   * - **EnergyShiftDetector**
     - Distribution comparison
     - Reference data
     - Energy distance between distributions
   * - **ClassifierShiftDetector / BBSDDetector**
     - Feature-rich detection
     - Reference data
     - Train a classifier to distinguish distributions; very powerful
   * - **LabelShiftDetector**
     - Label proportions changed
     - Reference data + model
     - Detects when class frequencies shift
   * - **ImportanceWeightingShift**
     - Covariate shift correction
     - Reference data
     - Estimates density ratio for reweighting

**Rule of thumb**: ``MMDShiftDetector`` for general-purpose monitoring. ``ClassifierShiftDetector`` when you need maximum sensitivity.

Decision Flowchart
------------------

::

   Do you need provable coverage guarantees?
   ├─ YES → Use Conformal Prediction (raps for classification, cv_plus for regression)
   └─ NO
      ├─ Are your model's confidence scores unreliable?
      │  ├─ YES → Start with Calibration (TemperatureScaling)
      │  └─ NO → Continue below
      ├─ Do you need to detect unknown/novel inputs?
      │  ├─ YES → Use OOD Detection (Energy or Mahalanobis)
      │  └─ NO → Continue below
      ├─ Should the model abstain on hard examples?
      │  ├─ YES → Use Selective Prediction (SoftmaxThreshold or SelectiveNet)
      │  └─ NO → Continue below
      ├─ Do you need epistemic vs aleatoric decomposition?
      │  ├─ YES → Use Bayesian Methods (DeepEnsemble or MCDropout)
      │  └─ NO → Continue below
      ├─ Is your data distribution changing over time?
      │  ├─ YES → Use Shift Detection (MMDShiftDetector)
      │  └─ NO → Continue below
      └─ Using LLMs? → Use LLM Uncertainty (SemanticEntropy, TokenEntropy)

Combining Methods
-----------------

Methods in incerto are designed to be composable. Common combinations:

1. **Calibration + Conformal**: Calibrate first, then apply conformal prediction for guaranteed coverage with tighter sets.

2. **OOD Detection + Selective Prediction**: Use OOD scores as a rejection criterion — abstain on OOD inputs.

3. **Bayesian + Active Learning**: Use epistemic uncertainty from ``MCDropout`` or ``DeepEnsemble`` as acquisition function for active learning.

4. **Shift Detection + Recalibration**: Monitor for shift with ``MMDShiftDetector``, trigger recalibration with ``TemperatureScaling`` when drift is detected.

5. **LLM Uncertainty + Selective Prediction**: Use ``SemanticEntropy`` to decide when to show a human the LLM output vs. present it directly.
