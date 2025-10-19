# NLP Examples

Uncertainty quantification for natural language processing and large language models (LLMs).

## Overview

Language models present unique challenges for uncertainty quantification:
- **Sequential predictions**: Uncertainty can be measured at token or sequence level
- **Autoregressive generation**: Each token depends on previous ones
- **Large vocabulary**: Thousands of possible outputs per token
- **Sampling-based methods**: Generate multiple outputs to estimate uncertainty

This directory demonstrates uncertainty methods specifically designed for LLMs.

---

## Examples

### llm_uncertainty.py
**Runtime**: ~5 minutes (with small model like Qwen2.5-0.5B)
**Difficulty**: ⭐⭐⭐

Comprehensive uncertainty quantification for language models.

**What it demonstrates**:

#### 1. Token-Level Uncertainty
Methods that operate on individual token predictions:

- **Token Entropy**: Uncertainty in next token distribution
  ```python
  entropy = TokenEntropy.compute(logits)  # shape: [batch, seq_len]
  ```

- **Token Confidence**: Max probability (inverse of entropy)
  ```python
  confidence = TokenConfidence.compute(logits)
  ```

- **Surprisal**: Negative log-probability of actual token
  ```python
  surprisal = SurprisalScore.compute(logits, token_ids)
  ```

- **Perplexity**: Exponential of average negative log-likelihood
  ```python
  perplexity = TokenPerplexity.compute(logits, token_ids)
  ```

#### 2. Sequence-Level Uncertainty
Aggregate token-level information to sequence level:

- **Sequence Probability**: Joint probability of sequence
  ```python
  seq_prob = SequenceProbability.compute(logits, token_ids)
  ```

- **Average Log Probability**: Mean log-prob over tokens
  ```python
  avg_log_prob = AverageLogProb.compute(logits, token_ids)
  ```

- **Sequence Entropy**: Aggregated entropy over sequence
  ```python
  seq_entropy = SequenceEntropy.compute(logits, aggregation='mean')
  ```

#### 3. Sampling-Based Uncertainty
Generate multiple outputs to estimate uncertainty:

- **Self-Consistency**: Agreement rate among multiple generations
  ```python
  responses = [generate(prompt) for _ in range(n_samples)]
  result = SelfConsistency.compute(responses)
  # Returns: agreement_rate, entropy, most_common_answer
  ```

- **Semantic Entropy**: Entropy over semantically clustered responses
  ```python
  result = SemanticEntropy.compute(responses)
  # Clusters similar responses, computes entropy over clusters
  ```

- **Predictive Entropy**: Expected entropy over generations
  ```python
  logit_samples = [model(prompt) for _ in range(n_samples)]
  pred_entropy = PredictiveEntropy.compute(logit_samples)
  ```

- **Mutual Information**: Epistemic uncertainty (model uncertainty)
  ```python
  mi = MutualInformation.compute(logit_samples)
  # High MI = model is uncertain, not just data noise
  ```

#### 4. Generation-Based Methods
Uncertainty during decoding:

- **Beam Search Uncertainty**: Entropy over beam hypotheses
  ```python
  result = BeamSearchUncertainty.compute_from_scores(beam_scores)
  ```

- **Nucleus Sampling**: Effective vocabulary size
  ```python
  eff_vocab = NucleusSamplingUncertainty.effective_vocabulary_size(logits, p=0.9)
  ```

- **I Don't Know Detection**: Detect uncertainty phrases
  ```python
  has_uncertainty = IDontKnowDetection.contains_uncertainty_phrase(text)
  ```

#### 5. Calibration for LLMs

- **Token Temperature Scaling**: Calibrate token-level probabilities
  ```python
  calibrator = TokenTemperatureScaling(init_temp=1.0)
  calibrator.fit(val_logits, val_token_ids)
  calibrated_logits = calibrator(test_logits)
  ```

- **Histogram Binning**: Non-parametric calibration
  ```python
  calibrator = HistogramBinning(n_bins=10)
  calibrator.fit(confidences, correctness)
  ```

#### 6. Evaluation Metrics

- **Selective Accuracy**: Accuracy when rejecting low-confidence
  ```python
  result = selective_accuracy(preds, targets, confidences, threshold=0.7)
  # Returns: accuracy, coverage, num_selected
  ```

- **Calibration Error**: ECE and MCE for token predictions
  ```python
  cal_metrics = calibration_error(confidences, correctness)
  # Returns: ece, mce
  ```

- **AUR-C**: Area under risk-coverage curve
  ```python
  aurc_score = aur_c(confidences, correctness)
  ```

**Key patterns**:

```python
# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Get logits
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Compute token-level uncertainty
entropy = TokenEntropy.compute(logits)
confidence = TokenConfidence.compute(logits)

# Compute sequence-level uncertainty
seq_prob = SequenceProbability.compute(logits, inputs.input_ids)

# Sampling-based uncertainty
responses = [
    model.generate(inputs.input_ids, do_sample=True, temperature=1.0)
    for _ in range(10)
]
self_consistency = SelfConsistency.compute(responses)
```

---

## When to Use Each Method

### Token-Level Methods
**Use when**:
- You want to identify uncertain tokens in a generation
- Building confidence-based beam search
- Token-by-token decision making

**Examples**:
- Highlight uncertain words in translations
- Pause generation when uncertainty is high
- Filter low-confidence tokens in summarization

### Sequence-Level Methods
**Use when**:
- You want a single uncertainty score for the whole output
- Ranking multiple candidate generations
- Binary accept/reject decisions

**Examples**:
- Rank multiple answers to a question
- Decide whether to show output to user
- Selective prediction for entire response

### Sampling-Based Methods
**Use when**:
- You can afford multiple generations (higher cost)
- You want to capture model uncertainty (epistemic)
- Ground truth is not available

**Examples**:
- Critical applications (medical, legal)
- Detecting hallucinations
- Ensemble-style uncertainty without training multiple models

### Generation-Based Methods
**Use when**:
- Uncertainty should guide the decoding process
- You want to detect "I don't know" responses
- Analyzing beam search diversity

**Examples**:
- Adaptive generation (stop early if low uncertainty)
- Force model to express uncertainty
- Analyze model's awareness of knowledge gaps

---

## Running the Example

```bash
# From repository root
python examples/04_nlp/llm_uncertainty.py
```

**Requirements**:
- `transformers` library
- A small model like Qwen2.5-0.5B (or modify to use GPT-2)
- ~2GB VRAM for Qwen2.5-0.5B
- Can run on CPU but slower

**Output**:
- Visualizations saved to `output/nlp/`
- Token-level uncertainty heatmaps
- Sequence-level distribution plots
- Self-consistency analysis
- Calibration curves

---

## Model Selection

### Recommended Models for Testing

**Small models (good for learning)**:
- `Qwen/Qwen2.5-0.5B` - Very small, fast
- `gpt2` - Classic, widely used
- `distilgpt2` - Even smaller

**Medium models (better quality)**:
- `Qwen/Qwen2.5-1.5B`
- `gpt2-medium`
- `EleutherAI/pythia-1.4b`

**Large models (production-like)**:
- `Qwen/Qwen2.5-7B` (requires more GPU)
- `meta-llama/Llama-2-7b` (requires access)

To change model:
```python
model_name = "gpt2"  # Change this line
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## Common Patterns

### Pattern 1: Token-Level Analysis
```python
# Get logits for text
inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits

# Compute multiple token-level metrics
metrics = {
    'entropy': TokenEntropy.compute(logits),
    'confidence': TokenConfidence.compute(logits),
    'perplexity': TokenPerplexity.compute(logits, inputs.input_ids),
}

# Visualize token uncertainty
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
plot_token_uncertainty(tokens, metrics['entropy'])
```

### Pattern 2: Self-Consistency for Question Answering
```python
question = "What is the capital of France?"
prompt = f"Question: {question}\nAnswer:"

# Generate multiple responses
n_samples = 10
responses = []
for _ in range(n_samples):
    output = model.generate(
        inputs.input_ids,
        max_length=50,
        do_sample=True,
        temperature=1.0
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append(response)

# Compute self-consistency
result = SelfConsistency.compute(responses)
print(f"Agreement rate: {result['agreement_rate']:.2f}")
print(f"Most common answer: {result['most_common']}")
print(f"Entropy: {result['entropy']:.4f}")

# High agreement = confident, low agreement = uncertain
```

### Pattern 3: Semantic Entropy for Hallucination Detection
```python
responses = [generate(prompt, temp=1.0) for _ in range(20)]

# Cluster semantically similar responses
result = SemanticEntropy.compute(responses, similarity_threshold=0.8)

print(f"Number of distinct semantic clusters: {result['num_clusters']}")
print(f"Semantic entropy: {result['semantic_entropy']:.4f}")

# High semantic entropy = many different answers = likely hallucinating
if result['semantic_entropy'] > 2.0:
    print("Warning: High uncertainty, possible hallucination")
```

### Pattern 4: Selective Prediction
```python
# Generate with confidence tracking
logits = model(**inputs).logits
probs = F.softmax(logits, dim=-1)
max_probs, predictions = probs.max(dim=-1)

# Token-level confidence
token_confidence = max_probs.mean()

# Only accept if confident
threshold = 0.7
if token_confidence > threshold:
    print(f"Confident (conf={token_confidence:.2f}): {generated_text}")
else:
    print(f"Uncertain (conf={token_confidence:.2f}): Abstaining")
```

---

## Tips for LLM Uncertainty

### 1. Temperature Matters
- Higher temperature → more diverse samples → better uncertainty estimation
- Use temperature=1.0 for uncertainty estimation
- Don't use temperature=0 (greedy) for uncertainty

### 2. Number of Samples
- Self-consistency: 5-10 samples minimum
- Semantic entropy: 10-20 samples
- Mutual information: 10-30 samples
- More samples = better estimates but higher cost

### 3. Prompt Engineering
Explicit uncertainty prompts:
```python
prompt = """Question: {question}
Answer with your confidence level (low/medium/high):"""
```

### 4. Combining Methods
Best practice: Use multiple methods
```python
# Token-level for generation quality
token_ent = TokenEntropy.compute(logits).mean()

# Sampling-based for epistemic uncertainty
self_cons = SelfConsistency.compute(responses)

# Combined decision
if token_ent < 1.0 and self_cons['agreement_rate'] > 0.7:
    accept_output()
else:
    reject_output()
```

### 5. Computational Cost
Methods ranked by cost (cheap → expensive):
1. Token confidence/entropy (single forward pass)
2. Sequence probability (single forward pass)
3. Beam search uncertainty (already computed in beam search)
4. Self-consistency (N generations)
5. Semantic entropy (N generations + embedding)
6. Mutual information (N forward passes)

---

## Common Questions

**Q: Which uncertainty method is best?**
A: Depends on use case:
- Quick single-pass: Token entropy
- Best epistemic: Mutual information or semantic entropy
- Question answering: Self-consistency
- Detection: I-don't-know detection

**Q: How many samples for self-consistency?**
A: Start with 5-10. Increase to 20+ for critical applications.

**Q: Can I use these with API models (GPT-4, Claude)?**
A: Only sampling-based methods (self-consistency, semantic entropy). No access to logits.

**Q: How to calibrate LLM confidence?**
A: Use validation set with known answers. Fit TemperatureScaling or HistogramBinning.

**Q: What's the difference between entropy and perplexity?**
A: Perplexity = exp(entropy). Both measure uncertainty, perplexity is more interpretable as "effective vocabulary size".

**Q: How to detect hallucinations?**
A: High semantic entropy across multiple samples indicates hallucination. Also combine with factuality checks.

---

## What's Next?

After understanding LLM uncertainty:
- Apply to your specific LLM tasks
- Experiment with different models
- Try on longer texts (summarization, articles)
- Implement uncertainty-aware decoding
- Build selective prediction systems

---

**Previous**: [Vision Examples](../03_vision/) or [Main Examples](../)
