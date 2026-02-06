"""
Tests for OOD detection visualization functions.
"""

import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from incerto.ood.visual import plot_roc, score_hist


class TestPlotROC:
    """Test ROC curve plotting."""

    def test_basic_plot(self):
        """Test basic ROC curve generation."""
        id_scores = torch.zeros(50)
        ood_scores = torch.ones(50)
        ax = plot_roc(id_scores, ood_scores)
        assert ax is not None
        plt.close()

    def test_with_label(self):
        """Test ROC curve with label."""
        id_scores = torch.randn(50)
        ood_scores = torch.randn(50) + 1.0
        ax = plot_roc(id_scores, ood_scores, label="Test Detector")
        assert ax is not None
        plt.close()

    def test_with_provided_axes(self):
        """Test plotting on provided axes."""
        fig, ax = plt.subplots()
        id_scores = torch.randn(30)
        ood_scores = torch.randn(30) + 1.0
        returned_ax = plot_roc(id_scores, ood_scores, ax=ax)
        assert returned_ax is ax
        plt.close()

    def test_returns_axes(self):
        """Should return matplotlib Axes object."""
        id_scores = torch.randn(20)
        ood_scores = torch.randn(20)
        ax = plot_roc(id_scores, ood_scores)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_different_sizes(self):
        """Should handle different ID and OOD sample sizes."""
        id_scores = torch.randn(30)
        ood_scores = torch.randn(70)
        ax = plot_roc(id_scores, ood_scores)
        assert ax is not None
        plt.close()


class TestScoreHist:
    """Test score histogram plotting."""

    def test_basic_plot(self):
        """Test basic histogram generation."""
        id_scores = torch.randn(100)
        ood_scores = torch.randn(100) + 2.0
        ax = score_hist(id_scores, ood_scores)
        assert ax is not None
        plt.close()

    def test_with_provided_axes(self):
        """Test plotting on provided axes."""
        fig, ax = plt.subplots()
        id_scores = torch.randn(50)
        ood_scores = torch.randn(50) + 1.0
        returned_ax = score_hist(id_scores, ood_scores, ax=ax)
        assert returned_ax is ax
        plt.close()

    def test_custom_bins(self):
        """Test with custom number of bins."""
        id_scores = torch.randn(100)
        ood_scores = torch.randn(100)
        ax = score_hist(id_scores, ood_scores, bins=20)
        assert ax is not None
        plt.close()

    def test_returns_axes(self):
        """Should return matplotlib Axes object."""
        id_scores = torch.randn(30)
        ood_scores = torch.randn(30)
        ax = score_hist(id_scores, ood_scores)
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_has_legend(self):
        """Histogram should have a legend."""
        id_scores = torch.randn(50)
        ood_scores = torch.randn(50) + 1.0
        ax = score_hist(id_scores, ood_scores)
        legend = ax.get_legend()
        assert legend is not None
        plt.close()

    def test_has_labels(self):
        """Histogram should have axis labels."""
        id_scores = torch.randn(50)
        ood_scores = torch.randn(50)
        ax = score_hist(id_scores, ood_scores)
        assert ax.get_xlabel() == "OOD score"
        assert ax.get_ylabel() == "# samples"
        plt.close()
