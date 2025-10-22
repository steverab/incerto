Selective Prediction Guide
==========================

Selective prediction (also called prediction with rejection) allows models to **abstain** from uncertain predictions, improving accuracy on predictions that are made.

Why Selective Prediction
-------------------------

Key idea: **Don't predict when uncertain**

Benefits:
   - Higher accuracy on accepted predictions
   - Explicit uncertainty communication
   - Safer deployment in critical applications

Trade-off:
   - **Coverage**: Fraction of samples where prediction is made
   - **Risk**: Error rate on predictions that are made

Goal: Maximize accuracy while maintaining acceptable coverage.

Basic Concepts
--------------

**Selective classifier**:
   (f, g) where f predicts, g decides whether to abstain

**Coverage**:
   Φ = P(g(x) = 1) = fraction of samples where we predict

**Selective risk**:
   R_φ = E[ℓ(f(x), y) | g(x) = 1] = error rate on accepted samples

**Selective accuracy**:
   Accuracy on samples where prediction is made

Methods
-------

Softmax Threshold
^^^^^^^^^^^^^^^^^

**Best for**: Simple baseline, post-hoc application

Abstain when max softmax probability < threshold:

.. code-block:: python

   from incerto.sp import SoftmaxThreshold

   selector = SoftmaxThreshold(threshold=0.9)

   # Make predictions
   logits = model(test_data)
   predictions, abstention = selector.predict(logits)

   # predictions[i] is None where abstention[i] is True
   coverage = (~abstention).float().mean()
   accepted_preds = predictions[~abstention]
   accepted_labels = labels[~abstention]

   accuracy = (accepted_preds == accepted_labels).float().mean()

   print(f"Coverage: {coverage:.2%}")
   print(f"Selective accuracy: {accuracy:.2%}")

**Advantages**:
   - Simple, fast
   - No retraining needed
   - Interpretable

**Disadvantages**:
   - Threshold requires tuning
   - May not be optimal

Monte Carlo Dropout
^^^^^^^^^^^^^^^^^^^

**Best for**: Bayesian uncertainty estimates

Use dropout uncertainty for selection:

.. code-block:: python

   from incerto.bayesian import MCDropout
   from incerto.sp import UncertaintyBasedSelection

   # Enable dropout during inference
   mc_dropout = MCDropout(model, n_samples=10)

   # Get predictions with uncertainty
   result = mc_dropout.predict(test_data)

   # Select based on epistemic uncertainty
   selector = UncertaintyBasedSelection(threshold=0.1)
   predictions, abstention = selector.predict(
       result['mean'],
       result['epistemic']
   )

Entropy-Based Selection
^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Information-theoretic approach

Abstain on high-entropy predictions:

.. code-block:: python

   from incerto.sp import EntropyThreshold

   selector = EntropyThreshold(threshold=0.5)

   logits = model(test_data)
   predictions, abstention = selector.predict(logits)

Self-Adaptive Training (SAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Training models for selective prediction

Train model to output selection scores:

.. code-block:: python

   from incerto.sp import SelfAdaptiveTraining

   model_with_selection = SelfAdaptiveTraining(
       base_model,
       num_classes=10
   )

   # Training loop
   for inputs, labels in train_loader:
       logits, selection_scores = model_with_selection(inputs)

       # SAT loss combines classification and selection
       loss = model_with_selection.sat_loss(
           logits,
           selection_scores,
           labels,
           coverage=0.8  # Target 80% coverage
       )

       loss.backward()
       optimizer.step()

   # Inference with learned selection
   logits, selection_scores = model_with_selection(test_data)
   predictions = logits.argmax(dim=-1)
   abstention = selection_scores < threshold

Complete Workflow
-----------------

.. code-block:: python

   import torch
   from incerto.sp import SoftmaxThreshold, selective_risk, coverage_rate

   # 1. Train model normally
   model = train_model(train_loader)

   # 2. Choose selection strategy
   selector = SoftmaxThreshold(threshold=0.95)

   # 3. Evaluate on validation set to choose threshold
   val_logits, val_labels = get_predictions(model, val_loader)
   predictions, abstention = selector.predict(val_logits)

   coverage = (~abstention).float().mean()
   accepted = ~abstention
   selective_acc = (predictions[accepted] == val_labels[accepted]).float().mean()

   print(f"Coverage: {coverage:.2%}")
   print(f"Selective accuracy: {selective_acc:.2%}")

   # 4. Adjust threshold to achieve desired coverage
   # Try different thresholds...

   # 5. Deploy with chosen threshold
   def predict_with_abstention(x):
       logits = model(x)
       pred, abstain = selector.predict(logits)

       if abstain:
           return None  # Abstain - defer to human
       return pred

Metrics
-------

**Coverage-Risk Curve**:
   Plot selective accuracy vs. coverage

.. code-block:: python

   from incerto.sp import plot_coverage_risk_curve

   thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
   coverages, risks = [], []

   for threshold in thresholds:
       selector = SoftmaxThreshold(threshold=threshold)
       preds, abstention = selector.predict(logits)

       coverage = (~abstention).float().mean()
       risk = selective_risk(preds[~abstention], labels[~abstention])

       coverages.append(coverage.item())
       risks.append(risk.item())

   # Plot
   import matplotlib.pyplot as plt
   plt.plot(coverages, risks)
   plt.xlabel('Coverage')
   plt.ylabel('Selective Risk')
   plt.title('Coverage-Risk Curve')

**Area Under Risk-Coverage Curve (AURC)**:
   Lower is better (perfect = 0)

.. code-block:: python

   from incerto.sp import aurc

   score = aurc(coverages, risks)

Best Practices
--------------

1. **Tune threshold on validation data**
      Never use test data for threshold selection

2. **Consider deployment constraints**
      What coverage rate is acceptable?

3. **Combine with calibration**
      Calibrated models have better selection

4. **Monitor in production**
      Track coverage and accuracy over time

5. **Plan for abstention**
      What happens when model abstains? (Human review, fallback model, etc.)

6. **Use multiple signals**
      Combine softmax, entropy, MC dropout for better selection

Trade-offs
----------

**High threshold** (e.g., 0.95):
   - Lower coverage (~70%)
   - Higher accuracy on accepted samples
   - More abstentions

**Low threshold** (e.g., 0.7):
   - Higher coverage (~95%)
   - Lower accuracy on accepted samples
   - Fewer abstentions

Choose based on:
   - Cost of errors vs. cost of abstention
   - Availability of fallback (human expert, simpler model)
   - Application requirements

References
----------

1. Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks" (NeurIPS 2017)
2. Geifman & El-Yaniv, "SelectiveNet: A Deep Neural Network with a Rejection Option" (ICML 2019)
3. Mozannar & Sontag, "Consistent Estimators for Learning to Defer" (NeurIPS 2020)

See Also
--------

- :doc:`../api/sp` - Complete API reference
- :doc:`calibration` - Calibration for better confidence
- :doc:`conformal` - Prediction sets with guarantees
- :doc:`ood` - OOD detection
