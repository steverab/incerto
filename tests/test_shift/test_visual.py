"""Tests for shift detection visualization functions."""

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from incerto.shift.visual import (
    plot_confidence_distributions,
    plot_embedding_space,
    plot_feature_histograms,
    plot_ks_statistics,
    plot_shift_severity,
)


@pytest.fixture
def ref_data():
    """Reference distribution samples."""
    torch.manual_seed(42)
    return torch.randn(50, 10)


@pytest.fixture
def test_data():
    """Test distribution samples (shifted)."""
    torch.manual_seed(43)
    return torch.randn(40, 10) + 1.0


class TestPlotFeatureHistograms:
    def test_returns_figure(self, ref_data, test_data):
        """Should return a matplotlib Figure."""
        fig = plot_feature_histograms(ref_data, test_data, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_features(self, ref_data, test_data):
        """Should plot first 5 features by default."""
        fig = plot_feature_histograms(ref_data, test_data, show=False)
        # Should have 5 axes (subplots)
        assert len(fig.axes) == 5
        plt.close(fig)

    def test_custom_features(self, ref_data, test_data):
        """Should plot specified features."""
        fig = plot_feature_histograms(ref_data, test_data, feature_ids=[0, 2, 5], show=False)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_single_feature(self, ref_data, test_data):
        """Should handle single feature."""
        fig = plot_feature_histograms(ref_data, test_data, feature_ids=[0], show=False)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_custom_bins(self, ref_data, test_data):
        """Should accept custom bin count."""
        fig = plot_feature_histograms(ref_data, test_data, bins=50, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fewer_features_than_default(self):
        """Should handle data with fewer than 5 features."""
        ref = torch.randn(50, 3)
        test = torch.randn(40, 3)
        fig = plot_feature_histograms(ref, test, show=False)
        assert len(fig.axes) == 3
        plt.close(fig)


class TestPlotEmbeddingSpace:
    def test_returns_figure_tsne(self, ref_data, test_data):
        """Should return a matplotlib Figure with t-SNE."""
        fig = plot_embedding_space(ref_data, test_data, method="tsne", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_pca(self, ref_data, test_data):
        """Should return a matplotlib Figure with PCA."""
        fig = plot_embedding_space(ref_data, test_data, method="pca", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_faster_than_tsne(self, ref_data, test_data):
        """PCA should work (and be faster, but we just test it works)."""
        fig = plot_embedding_space(ref_data, test_data, method="pca", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_method_raises(self, ref_data, test_data):
        """Should raise ValueError for unsupported methods."""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_embedding_space(ref_data, test_data, method="umap", show=False)

    def test_title_contains_method(self, ref_data, test_data):
        """Title should indicate the method used."""
        fig = plot_embedding_space(ref_data, test_data, method="pca", show=False)
        ax = fig.axes[0]
        assert "PCA" in ax.get_title()
        plt.close(fig)

        fig = plot_embedding_space(ref_data, test_data, method="tsne", show=False)
        ax = fig.axes[0]
        assert "TSNE" in ax.get_title()
        plt.close(fig)

    def test_different_sample_sizes(self):
        """Should handle different sample sizes."""
        ref = torch.randn(100, 5)
        test = torch.randn(30, 5)
        fig = plot_embedding_space(ref, test, method="pca", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_high_dimensional(self):
        """Should handle high-dimensional data."""
        ref = torch.randn(50, 100)
        test = torch.randn(50, 100)
        fig = plot_embedding_space(ref, test, method="pca", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotConfidenceDistributions:
    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        ref_conf = torch.rand(100)
        test_conf = torch.rand(80) * 0.8  # Shifted lower
        fig = plot_confidence_distributions(ref_conf, test_conf, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_sizes(self):
        """Should handle different sample sizes."""
        ref_conf = torch.rand(200)
        test_conf = torch.rand(50)
        fig = plot_confidence_distributions(ref_conf, test_conf, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_bins(self):
        """Should accept custom bin count."""
        ref_conf = torch.rand(100)
        test_conf = torch.rand(100)
        fig = plot_confidence_distributions(ref_conf, test_conf, bins=20, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_numpy_input(self):
        """Should accept numpy arrays."""
        import numpy as np

        ref_conf = np.random.rand(100)
        test_conf = np.random.rand(100)
        fig = plot_confidence_distributions(ref_conf, test_conf, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotShiftSeverity:
    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        severity = [0, 10, 20, 30, 45]
        scores = [0.1, 0.15, 0.25, 0.4, 0.6]
        fig = plot_shift_severity(severity, scores, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_thresholds(self):
        """Should show threshold lines."""
        severity = [0, 10, 20, 30]
        scores = [0.1, 0.2, 0.3, 0.5]
        fig = plot_shift_severity(
            severity, scores, warning_threshold=0.25, critical_threshold=0.4, show=False
        )
        assert isinstance(fig, plt.Figure)
        # Check legend has threshold labels
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Warning" in legend_texts
        assert "Critical" in legend_texts
        plt.close(fig)

    def test_custom_labels(self):
        """Should use custom axis labels."""
        severity = [0, 1, 2, 3]
        scores = [0.1, 0.2, 0.3, 0.4]
        fig = plot_shift_severity(
            severity,
            scores,
            severity_label="Time (days)",
            score_label="MMD Score",
            show=False,
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Time (days)"
        assert ax.get_ylabel() == "MMD Score"
        plt.close(fig)


class TestPlotKSStatistics:
    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        ks_stats = torch.rand(20)
        fig = plot_ks_statistics(ks_stats, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_k(self):
        """Should show only top K features."""
        ks_stats = torch.rand(50)
        fig = plot_ks_statistics(ks_stats, top_k=5, show=False)
        ax = fig.axes[0]
        # Should have 5 bars
        assert len(ax.patches) == 5
        plt.close(fig)

    def test_all_features(self):
        """Should show all features when top_k=None."""
        ks_stats = torch.rand(15)
        fig = plot_ks_statistics(ks_stats, top_k=None, show=False)
        ax = fig.axes[0]
        assert len(ax.patches) == 15
        plt.close(fig)

    def test_custom_feature_names(self):
        """Should use custom feature names."""
        ks_stats = torch.tensor([0.3, 0.1, 0.5])
        names = ["width", "height", "depth"]
        fig = plot_ks_statistics(ks_stats, feature_names=names, top_k=None, show=False)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Sorted by KS stat: depth (0.5), width (0.3), height (0.1)
        assert labels == ["depth", "width", "height"]
        plt.close(fig)

    def test_numpy_input(self):
        """Should accept numpy arrays."""
        import numpy as np

        ks_stats = np.random.rand(10)
        fig = plot_ks_statistics(ks_stats, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_list_input(self):
        """Should accept lists."""
        ks_stats = [0.1, 0.2, 0.15, 0.3, 0.05]
        fig = plot_ks_statistics(ks_stats, top_k=3, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
