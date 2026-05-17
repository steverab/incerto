"""
Tests for LLM verbalized uncertainty methods.
"""

import pytest

from incerto.llm import (
    BidirectionalConsistency,
    PTrue,
    SelfEvaluation,
    VerbalizedConfidence,
)


class TestVerbalizedConfidence:
    """Test verbalized confidence extraction."""

    def test_extract_percentage_percent_sign(self):
        """Test extraction of percentage with % sign."""
        assert VerbalizedConfidence.extract_percentage("I'm 85% confident") == pytest.approx(0.85)
        assert VerbalizedConfidence.extract_percentage("Confidence: 95%") == pytest.approx(0.95)
        assert VerbalizedConfidence.extract_percentage("75.5%") == pytest.approx(0.755)

    def test_extract_percentage_word(self):
        """Test extraction of percentage with 'percent' word."""
        assert VerbalizedConfidence.extract_percentage("I am 80 percent sure") == pytest.approx(
            0.80
        )
        assert VerbalizedConfidence.extract_percentage("85 percent confidence") == pytest.approx(
            0.85
        )

    def test_extract_percentage_fraction(self):
        """Test extraction of fraction notation."""
        assert VerbalizedConfidence.extract_percentage("My confidence is 90/100") == pytest.approx(
            0.90
        )

    def test_extract_percentage_decimal(self):
        """Test extraction doesn't misinterpret decimals > 1."""
        # "85.5" from "85.5%" should be normalized to 0.855
        result = VerbalizedConfidence.extract_percentage("85.5%")
        assert result == pytest.approx(0.855)

    def test_extract_percentage_clamping(self):
        """Test values are clamped to [0, 1]."""
        assert VerbalizedConfidence.extract_percentage("120%") == 1.0

    def test_extract_percentage_negative_ignored(self):
        """Test negative sign is ignored (regex matches digits only)."""
        # "-5%" matches "5%" because regex doesn't capture the minus sign
        assert VerbalizedConfidence.extract_percentage("-5%") == pytest.approx(0.05)

    def test_extract_percentage_no_match(self):
        """Test returns None when no percentage found."""
        assert VerbalizedConfidence.extract_percentage("The answer is Paris") is None
        assert VerbalizedConfidence.extract_percentage("I think so") is None

    def test_extract_percentage_confidence_word(self):
        """Test extraction with 'confidence' keyword."""
        assert VerbalizedConfidence.extract_percentage("My confidence is 75") == pytest.approx(0.75)

    def test_get_confidence_prompt(self):
        """Test prompt generation."""
        prompt = VerbalizedConfidence.get_confidence_prompt("What is 2+2?", "The answer is 4")
        assert "What is 2+2?" in prompt
        assert "The answer is 4" in prompt
        assert "confident" in prompt.lower()


class TestPTrue:
    """Test P(True) probability extraction."""

    def test_get_ptrue_prompt(self):
        """Test P(True) prompt generation."""
        prompt = PTrue.get_ptrue_prompt("What is the capital of France?", "Paris")
        assert "What is the capital of France?" in prompt
        assert "Paris" in prompt
        assert "probability" in prompt.lower()

    def test_extract_probability_decimal(self):
        """Test extraction of decimal probabilities."""
        assert PTrue.extract_probability("0.85") == pytest.approx(0.85)
        assert PTrue.extract_probability("The probability is 0.92") == pytest.approx(0.92)

    def test_extract_probability_percentage(self):
        """Test extraction and conversion of percentages."""
        assert PTrue.extract_probability("85%") == pytest.approx(0.85)
        assert PTrue.extract_probability("92 percent") == pytest.approx(0.92)

    def test_extract_probability_clamping(self):
        """Test values > 1 are treated as percentages and normalized."""
        # "1.5" is matched as decimal, treated as percentage (1.5 > 1), divided by 100
        assert PTrue.extract_probability("1.5") == pytest.approx(0.015)
        # "150%" is matched, normalized to 1.5, then treated as percentage -> 0.015, clamped to 0.015
        # Actually: 150% -> 150 -> /100 = 1.5 -> /100 again? No, let's trace: value=150, >1 so /100 = 1.5, clamp to 1.0
        assert PTrue.extract_probability("150%") == 1.0

    def test_extract_probability_no_match(self):
        """Test returns None when no probability found."""
        assert PTrue.extract_probability("I am quite confident") is None


class TestSelfEvaluation:
    """Test self-evaluation prompt generation."""

    def test_get_critique_prompt(self):
        """Test critique prompt generation."""
        prompt = SelfEvaluation.get_critique_prompt(
            "What is the speed of light?", "299,792,458 m/s"
        )
        assert "What is the speed of light?" in prompt
        assert "299,792,458 m/s" in prompt
        assert "evaluate" in prompt.lower() or "critically" in prompt.lower()
        assert "accurate" in prompt.lower() or "correct" in prompt.lower()


class TestBidirectionalConsistency:
    """Test bidirectional consistency methods."""

    def test_paraphrase_prompts(self):
        """Test paraphrase generation."""
        question = "What is 2+2?"
        paraphrases = BidirectionalConsistency.paraphrase_prompts(question)

        assert len(paraphrases) >= 2
        assert question in paraphrases  # Original should be included
        # All paraphrases should contain the original question content
        for p in paraphrases:
            assert "2+2" in p

    def test_compute_consistency_all_same(self):
        """Test consistency with all identical answers."""
        answers = ["4", "4", "4", "4"]
        consistency = BidirectionalConsistency.compute_consistency(answers)
        assert consistency == 1.0

    def test_compute_consistency_all_different(self):
        """Test consistency with all different answers."""
        answers = ["4", "5", "6", "7"]
        consistency = BidirectionalConsistency.compute_consistency(answers)
        assert consistency == 0.0

    def test_compute_consistency_partial(self):
        """Test consistency with some matching answers (token Jaccard default)."""
        answers = ["4", "4", "5"]
        # pairs: (4,4)=1, (4,5)=0, (4,5)=0  →  mean = 1/3
        consistency = BidirectionalConsistency.compute_consistency(answers)
        assert consistency == pytest.approx(1.0 / 3.0)

    def test_compute_consistency_partial_exact(self):
        """Legacy exact-match: 2 unique out of 3 → 0.5."""
        answers = ["4", "4", "5"]
        consistency = BidirectionalConsistency.compute_consistency(answers, match="exact")
        assert consistency == pytest.approx(0.5)

    def test_compute_consistency_token_overlap_with_long_text(self):
        """Token-overlap survives small phrasing differences."""
        answers = [
            "The capital of Japan is Tokyo.",
            "The capital of Japan is Tokyo. It is located in...",
            "Tokyo, also known as the City of Light.",
        ]
        c = BidirectionalConsistency.compute_consistency(answers)
        # All share "Tokyo" → non-zero overlap (old exact-match would give 0.0).
        assert c > 0.0

    def test_compute_consistency_single_answer(self):
        """Test consistency with single answer."""
        answers = ["4"]
        consistency = BidirectionalConsistency.compute_consistency(answers)
        assert consistency == 1.0

    def test_compute_consistency_two_same(self):
        """Test consistency with two identical answers."""
        answers = ["4", "4"]
        consistency = BidirectionalConsistency.compute_consistency(answers)
        assert consistency == 1.0

    def test_compute_consistency_two_different(self):
        """Test consistency with two different answers."""
        answers = ["4", "5"]
        consistency = BidirectionalConsistency.compute_consistency(answers)
        # 2 unique, so consistency = 1 - (2-1)/(2-1) = 0
        assert consistency == 0.0
