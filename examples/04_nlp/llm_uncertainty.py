"""
Comprehensive LLM Uncertainty Quantification Example

This example demonstrates the full range of uncertainty quantification methods
for Large Language Models using the incerto.llm module.

We'll use a small, accessible model (Qwen2.5-0.5B) to demonstrate:
1. Token-level uncertainty
2. Sequence-level uncertainty
3. Sampling-based uncertainty (multiple generations)
4. Generation-specific methods (beam search, nucleus sampling)
5. Verbalized uncertainty
6. Calibration
7. Metrics and visualizations

Usage:
    python examples/llm_uncertainty.py --model Qwen/Qwen2.5-0.5B-Instruct
    python examples/llm_uncertainty.py --model meta-llama/Llama-3.2-1B-Instruct
    python examples/llm_uncertainty.py --model google/gemma-2-2b-it

Requirements:
    pip install transformers torch accelerate
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import warnings

warnings.filterwarnings("ignore")

# Import incerto.llm methods
from incerto.llm import (
    # Token-level
    TokenEntropy,
    TokenConfidence,
    TokenPerplexity,
    SurprisalScore,
    TopKConfidence,
    # Sequence-level
    SequenceProbability,
    AverageLogProb,
    SequenceEntropy,
    SequencePerplexity,
    # Sampling-based
    SelfConsistency,
    SemanticEntropy,
    PredictiveEntropy,
    MutualInformation,
    # Generation-specific
    BeamSearchUncertainty,
    NucleusSamplingUncertainty,
    IDontKnowDetection,
    # Calibration
    TokenTemperatureScaling,
    HistogramBinning,
    # Metrics
    selective_accuracy,
    calibration_error,
    # Visualization
    plot_token_uncertainty,
    plot_confidence_vs_correctness,
    plot_generation_diversity,
    plot_uncertainty_distribution,
)


def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    console.print(f"[cyan]Loading model: {model_name}[/]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device in ["cuda", "mps"] else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"[green]✓ Model loaded successfully[/]")
    return model, tokenizer


def demonstrate_token_level_uncertainty(model, tokenizer, prompt, device, console):
    """Demonstrate token-level uncertainty methods."""
    console.print("\n[bold cyan]=== TOKEN-LEVEL UNCERTAINTY ===[/]")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate with output_scores to get logits
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    # Stack logits
    logits = torch.stack(outputs.scores, dim=1)  # (batch, seq_len, vocab_size)
    generated_tokens = outputs.sequences[:, inputs.input_ids.shape[1] :]

    # Compute token-level uncertainties
    entropies = TokenEntropy.compute(logits[0])
    confidences = TokenConfidence.compute(logits[0])
    surprisals = SurprisalScore.compute(logits, generated_tokens)
    top_k_conf = TopKConfidence.compute(logits[0], k=5)

    # Decode tokens
    tokens = [tokenizer.decode(t) for t in generated_tokens[0]]

    # Display results
    table = Table(title="Token-Level Uncertainty")
    table.add_column("Token", style="cyan")
    table.add_column("Entropy", style="yellow")
    table.add_column("Confidence", style="green")
    table.add_column("Surprisal", style="magenta")
    table.add_column("Top-5 Mass", style="blue")

    for i, token in enumerate(tokens[:100]):  # Show first 20 tokens
        table.add_row(
            token.replace("\n", "\\n"),
            f"{entropies[i].item():.3f}",
            f"{confidences[i].item():.3f}",
            f"{surprisals[0, i].item():.3f}",
            f"{top_k_conf[i].item():.3f}",
        )

    console.print(table)

    return logits, generated_tokens, tokens, entropies


def demonstrate_sequence_level_uncertainty(logits, generated_tokens, console):
    """Demonstrate sequence-level uncertainty methods."""
    console.print("\n[bold cyan]=== SEQUENCE-LEVEL UNCERTAINTY ===[/]")

    # Compute sequence-level metrics
    seq_prob = SequenceProbability.compute(logits, generated_tokens)
    avg_log_prob = AverageLogProb.compute(logits, generated_tokens)
    seq_entropy = SequenceEntropy.compute(logits, aggregation="mean")
    perplexity = SequencePerplexity.compute(logits, generated_tokens)

    # Display results
    table = Table(title="Sequence-Level Uncertainty")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Sequence Probability", f"{seq_prob[0].item():.6e}")
    table.add_row("Average Log Prob", f"{avg_log_prob[0].item():.4f}")
    table.add_row("Sequence Entropy (mean)", f"{seq_entropy[0].item():.4f}")
    table.add_row("Perplexity", f"{perplexity[0].item():.4f}")

    console.print(table)


def demonstrate_sampling_based_uncertainty(model, tokenizer, prompt, device, console):
    """Demonstrate sampling-based uncertainty with multiple generations."""
    console.print("\n[bold cyan]=== SAMPLING-BASED UNCERTAINTY ===[/]")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate multiple samples
    n_samples = 10
    responses = []
    all_logits = []

    console.print(f"Generating {n_samples} samples with temperature=0.8...")

    for i in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_text = tokenizer.decode(
            outputs.sequences[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        responses.append(generated_text)

        # Collect logits
        logits = torch.stack(outputs.scores, dim=0)  # (seq_len, vocab_size)
        all_logits.append(logits)

    # Self-consistency
    sc_result = SelfConsistency.compute(responses)

    # Semantic entropy
    sem_result = SemanticEntropy.compute(responses, similarity_threshold=0.7)

    # Predictive entropy
    pred_entropy = PredictiveEntropy.compute(all_logits)

    # Mutual information
    mi = MutualInformation.compute(all_logits)

    # Display results
    console.print(f"\n[yellow]Agreement Rate:[/] {sc_result['agreement_rate']:.2f}")
    console.print(f"[yellow]Response Entropy:[/] {sc_result['entropy']:.4f}")
    console.print(f"[yellow]Unique Responses:[/] {sc_result['num_unique']}/{n_samples}")
    console.print(f"[yellow]Semantic Clusters:[/] {sem_result['num_clusters']}")
    console.print(f"[yellow]Semantic Entropy:[/] {sem_result['semantic_entropy']:.4f}")
    console.print(
        f"[yellow]Mean Predictive Entropy:[/] {pred_entropy.mean().item():.4f}"
    )
    console.print(f"[yellow]Mean Mutual Information:[/] {mi.mean().item():.4f}")

    console.print("\n[bold]Sample responses:[/]")
    for i, response in enumerate(responses[:5], 1):
        console.print(f"  {i}. {response[:80]}...")

    return responses


def demonstrate_generation_specific(model, tokenizer, prompt, device, console):
    """Demonstrate generation-specific uncertainty methods."""
    console.print("\n[bold cyan]=== GENERATION-SPECIFIC UNCERTAINTY ===[/]")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            num_return_sequences=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Get beam scores
    beam_scores = (
        outputs.sequences_scores if hasattr(outputs, "sequences_scores") else None
    )

    if beam_scores is not None:
        beam_unc = BeamSearchUncertainty.compute_from_scores(beam_scores)
        console.print(f"[yellow]Beam Search Entropy:[/] {beam_unc['entropy']:.4f}")
        console.print(
            f"[yellow]Top Beam Probability:[/] {beam_unc['top_beam_prob']:.4f}"
        )

    # Nucleus sampling uncertainty
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    first_token_logits = outputs.scores[0][0]
    eff_vocab = NucleusSamplingUncertainty.effective_vocabulary_size(
        first_token_logits, p=0.9
    )
    prob_mass = NucleusSamplingUncertainty.probability_mass_concentration(
        first_token_logits, top_k=10
    )

    console.print(f"[yellow]Effective Vocabulary (p=0.9):[/] {eff_vocab}")
    console.print(f"[yellow]Top-10 Probability Mass:[/] {prob_mass:.4f}")


def demonstrate_verbalized_uncertainty(responses, console):
    """Demonstrate verbalized uncertainty detection."""
    console.print("\n[bold cyan]=== VERBALIZED UNCERTAINTY ===[/]")

    uncertain_count = 0
    hedging_count = 0

    for response in responses:
        # Check for "I don't know" phrases
        if IDontKnowDetection.contains_uncertainty_phrase(response):
            uncertain_count += 1

        # Check for hedging
        hedging_result = IDontKnowDetection.extract_confidence_from_hedging(response)
        if hedging_result["contains_hedging"]:
            hedging_count += 1

    console.print(
        f"[yellow]Responses with uncertainty phrases:[/] {uncertain_count}/{len(responses)}"
    )
    console.print(
        f"[yellow]Responses with hedging language:[/] {hedging_count}/{len(responses)}"
    )


def demonstrate_calibration(model, tokenizer, qa_pairs, device, console, save_dir):
    """Demonstrate calibration on Q&A pairs."""
    console.print("\n[bold cyan]=== CALIBRATION ===[/]")

    all_confidences = []
    all_correctness = []

    for question, true_answer in qa_pairs:
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        # Get generated answer
        generated = (
            tokenizer.decode(
                outputs.sequences[0, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            .strip()
            .lower()
        )

        # Get confidence (average token confidence)
        logits = torch.stack(outputs.scores, dim=1)
        confidences = TokenConfidence.compute(logits[0])
        avg_confidence = confidences.mean().item()

        # Check correctness (simple contains check)
        is_correct = true_answer.lower() in generated

        all_confidences.append(avg_confidence)
        all_correctness.append(1.0 if is_correct else 0.0)

    confidences_tensor = torch.tensor(all_confidences)
    correctness_tensor = torch.tensor(all_correctness)

    # Compute calibration metrics
    cal_err = calibration_error(confidences_tensor, correctness_tensor, n_bins=5)

    console.print(f"[yellow]Expected Calibration Error (ECE):[/] {cal_err['ece']:.4f}")
    console.print(f"[yellow]Maximum Calibration Error (MCE):[/] {cal_err['mce']:.4f}")

    # Plot calibration
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confidence_vs_correctness(
        confidences_tensor.numpy(), correctness_tensor.numpy(), n_bins=5, ax=ax
    )
    plt.savefig(save_dir / "llm_calibration.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]✓ Saved calibration plot to {save_dir / 'llm_calibration.png'}[/]"
    )
    plt.close()


def visualize_token_uncertainty(tokens, entropies, save_dir, console):
    """Visualize token-level uncertainty."""
    console.print("\n[bold cyan]=== VISUALIZATIONS ===[/]")

    # Token uncertainty heatmap
    fig, ax = plt.subplots(figsize=(12, 2))
    plot_token_uncertainty(
        tokens[:100],  # First 100 tokens
        entropies[:100].cpu().numpy(),
        ax=ax,
        title="Token-Level Uncertainty",
    )
    plt.savefig(save_dir / "token_uncertainty.png", dpi=150, bbox_inches="tight")
    console.print(
        f"[green]✓ Saved token uncertainty plot to {save_dir / 'token_uncertainty.png'}[/]"
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="LLM Uncertainty Quantification Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-0.5B-Instruct, meta-llama/Llama-3.2-1B-Instruct)",
    )

    # Device selection
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )

    args = parser.parse_args()

    # Setup
    global console
    console = Console()
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    console.print("[bold green]LLM Uncertainty Quantification - Incerto Library[/]")
    console.print(f"Device: {args.device}\n")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Example prompt
    # prompt = "Explain the concept of entropy in thermodynamics."
    prompt = "What are the best archer civs in AoE2? Respond with the top 5 civs. No explanation needed, just list the names."
    console.print(f"\n[bold]Prompt:[/] {prompt}")

    # 1. Token-level uncertainty
    logits, generated_tokens, tokens, entropies = demonstrate_token_level_uncertainty(
        model, tokenizer, prompt, args.device, console
    )

    # 2. Sequence-level uncertainty
    demonstrate_sequence_level_uncertainty(logits, generated_tokens, console)

    # 3. Sampling-based uncertainty
    responses = demonstrate_sampling_based_uncertainty(
        model, tokenizer, prompt, args.device, console
    )

    # 4. Generation-specific methods
    demonstrate_generation_specific(model, tokenizer, prompt, args.device, console)

    # 5. Verbalized uncertainty
    demonstrate_verbalized_uncertainty(responses, console)

    # 6. Calibration (with sample Q&A pairs)
    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is 2+2?", "4"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the speed of light?", "299,792,458 m/s"),
        ("What is the largest planet?", "Jupiter"),
    ]
    demonstrate_calibration(model, tokenizer, qa_pairs, args.device, console, save_dir)

    # 7. Visualizations
    visualize_token_uncertainty(tokens, entropies, save_dir, console)

    console.print("\n[bold green]✅ All demonstrations complete![/]")
    console.print(f"[green]Check {save_dir} for visualizations.[/]")


if __name__ == "__main__":
    main()
