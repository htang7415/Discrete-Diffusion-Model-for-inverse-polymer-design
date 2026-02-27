"""Plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from collections import Counter
from matplotlib.ticker import MaxNLocator


class PlotUtils:
    """Utility class for creating standardized plots."""

    def __init__(
        self,
        figure_size: Tuple[float, float] = (4.5, 4.5),
        font_size: int = 16,
        dpi: int = 150
    ):
        """Initialize plotting utilities.

        Args:
            figure_size: Default figure size (width, height).
            font_size: Default font size.
            dpi: DPI for saving figures.
        """
        self.figure_size = figure_size
        self.font_size = font_size
        self.dpi = dpi

        # Set global matplotlib parameters
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
            'figure.figsize': figure_size,
            'figure.dpi': dpi,
        })

    def histogram(
        self,
        data: List[np.ndarray],
        labels: List[str],
        xlabel: str,
        ylabel: str = "Frequency",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        bins: int = 50,
        alpha: float = 0.5,
        density: bool = False,
        style: str = 'stepfilled'
    ) -> plt.Figure:
        """Create overlaid histogram plot with clear distinction between distributions.

        Args:
            data: List of data arrays to plot.
            labels: List of labels for each data array.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save the figure.
            bins: Number of histogram bins.
            alpha: Transparency of bars.
            density: Whether to normalize histogram.
            style: Histogram style ('stepfilled', 'step', 'bar').

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Use contrasting colors that are easy to distinguish
        contrasting_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        colors = contrasting_colors[:len(data)]

        # Compute common bin edges for fair comparison
        all_data = np.concatenate([np.asarray(d) for d in data])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)

        for d, label, color in zip(data, labels, colors):
            if style == 'step':
                # Step histogram with thick lines, no fill
                ax.hist(d, bins=bin_edges, histtype='step', linewidth=2,
                        label=label, color=color, density=density)
            elif style == 'stepfilled':
                # Filled step histogram with edge
                ax.hist(d, bins=bin_edges, histtype='stepfilled', alpha=alpha,
                        label=label, color=color, density=density,
                        edgecolor=color, linewidth=1.5)
            else:
                # Traditional bar histogram
                ax.hist(d, bins=bin_edges, alpha=alpha, label=label, color=color,
                        density=density, edgecolor='white', linewidth=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        return fig

    def property_distribution_plot(
        self,
        predictions: List[float],
        target_value: float,
        xlabel: str,
        ylabel: str = "Count",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        bins: int = 50,
        color: str = "steelblue"
    ) -> plt.Figure:
        """Plot property distribution with a target reference line.

        Args:
            predictions: Predicted property values.
            target_value: Target property value to mark.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save the figure.
            bins: Number of histogram bins.
            color: Histogram color.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        if predictions:
            ax.hist(
                predictions,
                bins=bins,
                color=color,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5
            )
        else:
            span = max(1.0, abs(target_value) * 0.1)
            ax.set_xlim(target_value - span, target_value + span)
            ax.text(
                0.5,
                0.5,
                "No valid samples",
                transform=ax.transAxes,
                ha="center",
                va="center"
            )

        ax.axvline(
            target_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Target = {target_value}"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

        return fig

    def loss_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        xlabel: str = "Step",
        ylabel: str = "Loss",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        log_scale: bool = False
    ) -> plt.Figure:
        """Create loss curve plot.

        Args:
            train_losses: Training loss values.
            val_losses: Validation loss values.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save the figure.
            log_scale: Whether to use log scale for y-axis.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        steps = np.arange(1, len(train_losses) + 1)
        ax.plot(steps, train_losses, label='Train', color='blue')

        if val_losses:
            val_steps = np.arange(1, len(val_losses) + 1)
            # Adjust for different lengths (e.g., eval every N steps)
            if len(val_losses) < len(train_losses):
                ratio = len(train_losses) // len(val_losses)
                val_steps = val_steps * ratio
            ax.plot(val_steps, val_losses, label='Validation', color='orange')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        return fig

    def parity_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        xlabel: str = "True Value",
        ylabel: str = "Predicted Value",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> plt.Figure:
        """Create parity plot (true vs predicted).

        Args:
            y_true: True values.
            y_pred: Predicted values.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save the figure.
            metrics: Dictionary of metrics to annotate (e.g., MAE, RMSE, R2).

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none')

        # Add y = x line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        margin = (max_val - min_val) * 0.05
        line_range = [min_val - margin, max_val + margin]
        ax.plot(line_range, line_range, 'r--', linewidth=1.5, label='y = x')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        ax.set_xlim(line_range)
        ax.set_ylim(line_range)
        ax.set_aspect('equal')

        # Add metrics annotation
        if metrics:
            text_lines = [f"{k}: {v:.3f}" for k, v in metrics.items()]
            text = '\n'.join(text_lines)
            ax.text(
                0.05, 0.95, text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=self.font_size,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        return fig

    def bar_chart(
        self,
        categories: List[str],
        values: List[float],
        xlabel: Optional[str] = None,
        ylabel: str = "Value",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        color: str = 'steelblue'
    ) -> plt.Figure:
        """Create bar chart.

        Args:
            categories: Category labels.
            values: Values for each category.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save the figure.
            color: Bar color.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        x = np.arange(len(categories))
        ax.bar(x, values, color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')

        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        return fig

    def star_count_bar(
        self,
        star_counts: List[int],
        expected_count: int = 2,
        xlabel: str = "Star Count",
        ylabel: str = "Count",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        min_count: int = 0,
        max_count: Optional[int] = None,
        highlight_color: str = '#1f77b4',
        other_color: str = '#d0d0d0',
        annotate: bool = True
    ) -> plt.Figure:
        """Create star count distribution bar chart with expected-count highlight."""
        fig, ax = plt.subplots(figsize=self.figure_size)

        counter = Counter(star_counts)

        if not counter:
            counter[expected_count] = 0

        max_star = max(counter.keys())
        if max_count is not None:
            max_star = max(max_star, max_count)
        max_star = max(max_star, expected_count)

        min_star = min(min_count, expected_count, min(counter.keys()))

        categories = list(range(min_star, max_star + 1))
        values = [counter.get(k, 0) for k in categories]

        colors = [
            highlight_color if k == expected_count else other_color
            for k in categories
        ]
        x = np.arange(len(categories))
        ax.bar(
            x,
            values,
            color=colors,
            alpha=0.9,
            edgecolor='white',
            linewidth=0.7
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in categories])

        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        max_val = max(values) if values else 0

        if annotate and max_val > 0:
            pad = max(max_val * 0.02, 1)
            for xi, val in zip(x, values):
                if val == 0:
                    continue
                ax.text(
                    xi,
                    val + pad,
                    f"{int(val):,}",
                    ha='center',
                    va='bottom',
                    fontsize=self.font_size
                )
            ax.set_ylim(0, max_val + pad * 6)

        total = sum(values)
        if total > 0:
            expected_hits = counter.get(expected_count, 0)
            share = 100.0 * expected_hits / total
            summary = (
                f"Total: {total:,}\n"
                f"Star={expected_count}: {expected_hits:,} ({share:.1f}%)"
            )
            ax.text(
                0.98,
                0.95,
                summary,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=self.font_size,
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.85,
                    edgecolor='none'
                )
            )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        return fig

    def calibration_plot(
        self,
        target_values: List[float],
        mean_predictions: List[float],
        std_predictions: Optional[List[float]] = None,
        xlabel: str = "Target Value",
        ylabel: str = "Mean Predicted Value",
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create calibration plot for inverse design.

        Args:
            target_values: Target property values.
            mean_predictions: Mean predicted values for hits.
            std_predictions: Standard deviation of predictions.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save the figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        if std_predictions:
            ax.errorbar(
                target_values, mean_predictions, yerr=std_predictions,
                fmt='o', capsize=3, capthick=1, color='steelblue', alpha=0.8
            )
        else:
            ax.scatter(target_values, mean_predictions, color='steelblue', alpha=0.8)

        # Add y = x line
        min_val = min(min(target_values), min(mean_predictions))
        max_val = max(max(target_values), max(mean_predictions))
        margin = (max_val - min_val) * 0.1
        line_range = [min_val - margin, max_val + margin]
        ax.plot(line_range, line_range, 'r--', linewidth=1.5, label='y = x')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        return fig
