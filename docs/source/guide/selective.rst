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

   # Wrap your trained model
   selector = SoftmaxThreshold(model)
   selector.eval()

   # Forward pass with confidence scores
   with torch.no_grad():
       logits, confidence = selector(test_data, return_confidence=True)

   predictions = logits.argmax(dim=-1)

   # Set threshold and reject low-confidence samples
   threshold = confidence.quantile(0.2)  # reject bottom 20%
   rejected = selector.reject(confidence, threshold)
   selected = ~rejected

   accuracy = (predictions[selected] == labels[selected]).float().mean()
   coverage = selected.float().mean()

   print(f"Coverage: {coverage:.2%}")
   print(f"Selective accuracy: {accuracy:.2%}")

**Advantages**:
   - Simple, fast
   - No retraining needed
   - Interpretable

**Disadvantages**:
   - Threshold requires tuning
   - May not be optimal

Deep Gambler
^^^^^^^^^^^^

**Best for**: Learning when to abstain during training

Adds an extra abstain logit and trains with the gambler's loss:

.. code-block:: python

   from incerto.sp import DeepGambler

   # Create model with abstain head
   gambler = DeepGambler(backbone, num_classes=10, num_features=128)

   # Training loop
   for inputs, labels in train_loader:
       logits = gambler(inputs)  # shape: (batch, num_classes + 1)
       loss = gambler.gambler_loss(logits, labels, reward=2.2)
       loss.backward()
       optimizer.step()

   # Inference — confidence is 1 - P(abstain)
   logits, confidence = gambler(test_data, return_confidence=True)

Self-Adaptive Training (SAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Improving calibration during training for better selective prediction

Train model with adaptive soft labels that blend ground truth and model predictions:

.. code-block:: python

   from incerto.sp import SelfAdaptiveTraining

   sat = SelfAdaptiveTraining(
       backbone,
       num_classes=10,
       alpha_start=0.0,
       alpha_end=0.9,
       warmup_epochs=5,
   )

   # Training loop
   for epoch in range(total_epochs):
       alpha = sat.get_alpha(epoch, total_epochs)

       for inputs, labels in train_loader:
           logits = sat(inputs)
           loss = sat.sat_loss(logits, labels, alpha)
           loss.backward()
           optimizer.step()

   # Inference — uses MSP confidence like SoftmaxThreshold
   logits, confidence = sat(test_data, return_confidence=True)

SelectiveNet
^^^^^^^^^^^^^

**Best for**: Learning a dedicated selection function

Adds a selection head g(x) that outputs a selection probability:

.. code-block:: python

   from incerto.sp import SelectiveNet

   snet = SelectiveNet(backbone, num_classes=10, num_features=128)

   # Training loop — use the SelectiveNet loss
   for inputs, labels in train_loader:
       logits, selection = snet(inputs, return_confidence=True)
       loss = snet.selective_loss(logits, labels, selection, coverage_target=0.8)
       loss.backward()
       optimizer.step()

   # Inference — confidence comes from the selection head g(x)
   logits, confidence = snet(test_data, return_confidence=True)
   rejected = snet.reject(confidence, threshold=0.5)

Complete Workflow
-----------------

.. code-block:: python

   import torch
   from incerto.sp import SoftmaxThreshold, coverage, risk, aurc

   # 1. Train model normally
   model = train_model(train_loader)

   # 2. Wrap with selective predictor
   selector = SoftmaxThreshold(model)
   selector.eval()

   # 3. Get predictions and confidence on validation set
   with torch.no_grad():
       logits, confidence = selector(val_data, return_confidence=True)
   predictions = logits.argmax(dim=-1)

   # 4. Evaluate at different thresholds
   for threshold in [0.7, 0.8, 0.9, 0.95]:
       rejected = selector.reject(confidence, threshold)
       selected = ~rejected

       cov = coverage(rejected)
       sel_acc = (predictions[selected] == val_labels[selected]).float().mean()

       print(f"Threshold {threshold}: coverage={cov:.2%}, accuracy={sel_acc:.2%}")

   # 5. Compute AURC
   sorted_conf, idx = confidence.sort(descending=True)
   sorted_errors = (predictions[idx] != val_labels[idx]).float()
   score = aurc(sorted_conf, sorted_errors)
   print(f"AURC: {score:.4f}")

Metrics
-------

**Coverage-Risk Curve**:
   Plot risk vs. coverage across thresholds

.. code-block:: python

   from incerto.sp import plot_risk_coverage

   fig, ax = plt.subplots()
   plot_risk_coverage(logits, labels, confidence, ax=ax, show_aurc=True)

**Area Under Risk-Coverage Curve (AURC)**:
   Lower is better (perfect = 0)

.. code-block:: python

   from incerto.sp import aurc

   sorted_conf, idx = confidence.sort(descending=True)
   sorted_errors = (predictions[idx] != labels[idx]).float()
   score = aurc(sorted_conf, sorted_errors)

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

1. Chow, "An optimum character recognition system using decision functions" (1957)
2. Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks" (NeurIPS 2017)
3. Geifman & El-Yaniv, "SelectiveNet: A Deep Neural Network with a Rejection Option" (ICML 2019)
4. Ziyin et al., "Deep Gamblers: Learning to Abstain with Portfolio Theory" (NeurIPS 2019)
5. Huang et al., "Self-Adaptive Training: beyond Empirical Risk Minimization" (NeurIPS 2020)

See Also
--------

- :doc:`../api/sp` - Complete API reference
- :doc:`calibration` - Calibration for better confidence
- :doc:`conformal` - Prediction sets with guarantees
- :doc:`ood` - OOD detection
