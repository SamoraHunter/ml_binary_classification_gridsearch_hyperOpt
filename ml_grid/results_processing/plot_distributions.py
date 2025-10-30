# plot_distributions.py
"""
Distribution plotting module for ML results analysis.
Focuses on metric distributions with outcome variable stratification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
import warnings
import logging
from ml_grid.results_processing.core import get_clean_data, stratify_by_outcome

# Maximum number of outcomes to display in stratified plots to avoid clutter.
MAX_OUTCOMES_FOR_STRATIFIED_PLOT = 20
MAX_OUTCOMES_FOR_HEATMAP = 25


class DistributionPlotter:
    """A class for plotting metric distributions with outcome stratification support."""

    def __init__(self, data: pd.DataFrame, style: str = "default"):
        """Initializes the DistributionPlotter.

        Args:
            data (pd.DataFrame): A DataFrame containing the experiment results.
            style (str, optional): The matplotlib style to use for plots.
                Defaults to 'default'.
        """
        self.data = data
        self.clean_data = get_clean_data(data)
        self.logger = logging.getLogger("ml_grid")
        plt.style.use(style)

        # Set color palette
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        sns.set_palette("husl")

    def plot_metric_distributions(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        stratify_by_outcome: bool = False,
        outcomes_to_plot: Optional[List[str]] = None,
    ):
        """Plots distributions of key performance metrics.

        Args:
            metrics (Optional[List[str]], optional): A list of metric columns
                to plot. If None, uses a default list. Defaults to None.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (15, 10).
            stratify_by_outcome (bool, optional): If True, creates separate
                plots for each outcome. Defaults to False.
            outcomes_to_plot (Optional[List[str]], optional): A list of specific
                outcomes to plot. If None, all are used. Defaults to None.

        Raises:
            ValueError: If no specified metrics are found in the data.
        """
        if metrics is None:
            metrics = ["auc", "mcc", "f1", "precision", "recall", "accuracy"]

        available_metrics = [col for col in metrics if col in self.clean_data.columns]

        if not available_metrics:
            raise ValueError("No specified metrics found in data")

        if not stratify_by_outcome:
            self._plot_single_distribution(available_metrics, figsize)
        else:
            self._plot_stratified_distributions(
                available_metrics, figsize, outcomes_to_plot
            )

    def _plot_single_distribution(self, metrics: List[str], figsize: Tuple[int, int]):
        """Helper to plot distributions for all data combined."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            clean_values = self.clean_data[metric].dropna()

            ax.hist(
                clean_values, bins=30, alpha=0.7, edgecolor="black", color="skyblue"
            )
            ax.set_title(
                f"{metric.upper()} Distribution\n(n={len(clean_values)})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(metric.upper())
            ax.set_ylabel("Frequency")

            # Add statistics
            mean_val = clean_values.mean()
            median_val = clean_values.median()
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.3f}",
            )
            ax.axvline(
                median_val,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.3f}",
            )
            ax.legend(fontsize=9)

            # Add text box with additional stats
            stats_text = f"Std: {clean_values.std():.3f}\nMin: {clean_values.min():.3f}\nMax: {clean_values.max():.3f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "Metric Distributions - All Outcomes Combined",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def _plot_stratified_distributions(
        self,
        metrics: List[str],
        figsize: Tuple[int, int],
        outcomes_to_plot: Optional[List[str]] = None,
    ):
        """Helper to plot distributions stratified by outcome variable."""
        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found for stratification")

        outcomes = outcomes_to_plot or sorted(
            self.clean_data["outcome_variable"].unique()
        )
        if len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT:
            warnings.warn(
                f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Displaying the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                "Use the 'outcomes_to_plot' parameter to select specific outcomes.",
                stacklevel=2,
            )
            outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        n_outcomes = len(outcomes)
        n_metrics = len(metrics)

        # Create subplots
        fig, axes = plt.subplots(
            n_outcomes, n_metrics, figsize=(n_metrics * 4, n_outcomes * 3)
        )

        # Handle single metric or single outcome cases
        if n_outcomes == 1 and n_metrics == 1:
            axes = [[axes]]
        elif n_outcomes == 1:
            axes = [axes]
        elif n_metrics == 1:
            axes = [[ax] for ax in axes]

        colors = plt.cm.tab10(np.linspace(0, 1, n_outcomes))

        for i, outcome in enumerate(outcomes):
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]

            for j, metric in enumerate(metrics):
                ax = axes[i][j]

                if metric in outcome_data.columns:
                    clean_values = outcome_data[metric].dropna()

                    if len(clean_values) > 0:
                        ax.hist(
                            clean_values,
                            bins=20,
                            alpha=0.7,
                            edgecolor="black",
                            color=colors[i],
                        )

                        mean_val = clean_values.mean()
                        ax.axvline(
                            mean_val,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label=f"Mean: {mean_val:.3f}",
                        )

                        ax.set_title(
                            f"{outcome}\n{metric.upper()} (n={len(clean_values)})",
                            fontsize=10,
                            fontweight="bold",
                        )
                        ax.set_xlabel(metric.upper())
                        ax.set_ylabel("Frequency")
                        ax.legend(fontsize=8)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No Data",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                        ax.set_title(f"{outcome}\n{metric.upper()}", fontsize=10)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"{metric} not found",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                    ax.set_title(f"{outcome}\n{metric.upper()}", fontsize=10)

        plt.suptitle(
            "Metric Distributions by Outcome Variable", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def plot_comparative_distributions(
        self,
        metric: str = "auc",
        outcomes_to_compare: Optional[List[str]] = None,
        plot_type: str = "overlay",
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Creates comparative distribution plots across different outcomes.

        Args:
            metric (str, optional): The metric to compare. Defaults to 'auc'.
            outcomes_to_compare (Optional[List[str]], optional): A list of
                specific outcomes to compare. If None, all are used.
                Defaults to None.
            plot_type (str, optional): The type of plot to generate: 'overlay',
                'subplot', or 'violin'. Defaults to 'overlay'.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (12, 6).

        Raises:
            ValueError: If 'outcome_variable' or the specified metric is not
                found, or if an invalid `plot_type` is provided.
        """
        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found")

        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        outcomes = outcomes_to_compare or sorted(
            self.clean_data["outcome_variable"].unique()
        )

        if len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT:
            warnings.warn(
                f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Displaying the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                "Use the 'outcomes_to_compare' parameter to select specific outcomes.",
                stacklevel=2,
            )
            outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        if plot_type == "overlay":
            self._plot_overlay_distributions(metric, outcomes, figsize)
        elif plot_type == "subplot":
            self._plot_subplot_distributions(metric, outcomes, figsize)
        elif plot_type == "violin":
            self._plot_violin_distributions(metric, outcomes, figsize)
        else:
            raise ValueError("plot_type must be 'overlay', 'subplot', or 'violin'")

    def _plot_overlay_distributions(
        self, metric: str, outcomes: List[str], figsize: Tuple[int, int]
    ):
        """Helper to create an overlaid distribution plot."""
        plt.figure(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(outcomes)))

        for i, outcome in enumerate(outcomes):
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]
            clean_values = outcome_data[metric].dropna()

            if len(clean_values) > 0:
                plt.hist(
                    clean_values,
                    bins=20,
                    alpha=0.6,
                    label=f"{outcome} (n={len(clean_values)})",
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.5,
                )

        plt.xlabel(metric.upper(), fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(
            f"{metric.upper()} Distribution Comparison Across Outcomes",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_subplot_distributions(
        self, metric: str, outcomes: List[str], figsize: Tuple[int, int]
    ):
        """Helper to create subplot distributions."""
        n_outcomes = len(outcomes)
        cols = min(3, n_outcomes)
        rows = (n_outcomes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n_outcomes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, len(outcomes)))

        for i, outcome in enumerate(outcomes):
            ax = axes[i]
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]
            clean_values = outcome_data[metric].dropna()

            if len(clean_values) > 0:
                ax.hist(
                    clean_values,
                    bins=15,
                    alpha=0.7,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.5,
                )

                mean_val = clean_values.mean()
                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.3f}",
                )

                ax.set_title(
                    f"{outcome}\n(n={len(clean_values)})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel(metric.upper())
                ax.set_ylabel("Frequency")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"{metric.upper()} Distributions by Outcome", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def _plot_violin_distributions(
        self, metric: str, outcomes: List[str], figsize: Tuple[int, int]
    ):
        """Helper to create a violin plot for distribution comparison."""
        # Prepare data for violin plot
        plot_data = []

        for outcome in outcomes:
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]
            clean_values = outcome_data[metric].dropna()

            for value in clean_values:
                plot_data.append({"outcome": outcome, metric: value})

        plot_df = pd.DataFrame(plot_data)

        plt.figure(figsize=figsize)

        # Create violin plot
        sns.violinplot(data=plot_df, x="outcome", y=metric, inner="box")

        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Outcome Variable", fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(
            f"{metric.upper()} Distribution Comparison (Violin Plot)",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_distribution_summary_table(
        self, metrics: List[str] = None, outcomes_to_include: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Creates a summary table of distribution statistics by outcome.

        Args:
            metrics (Optional[List[str]], optional): A list of metrics to
                include in the summary. If None, uses a default list.
                Defaults to None.
            outcomes_to_include (Optional[List[str]], optional): A list of
                specific outcomes to include. If None, all are used.
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the summary statistics.
        Raises:
            ValueError: If 'outcome_variable' column is not found.
        """
        if metrics is None:
            metrics = ["auc", "mcc", "f1", "precision", "recall", "accuracy"]

        available_metrics = [col for col in metrics if col in self.clean_data.columns]

        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found")

        outcomes = outcomes_to_include or sorted(
            self.clean_data["outcome_variable"].unique()
        )

        summary_data = []

        for outcome in outcomes:
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]

            row = {"outcome_variable": outcome, "sample_size": len(outcome_data)}

            for metric in available_metrics:
                clean_values = outcome_data[metric].dropna()
                if len(clean_values) > 0:
                    row[f"{metric}_mean"] = clean_values.mean()
                    row[f"{metric}_std"] = clean_values.std()
                    row[f"{metric}_median"] = clean_values.median()
                    row[f"{metric}_min"] = clean_values.min()
                    row[f"{metric}_max"] = clean_values.max()
                else:
                    for stat in ["mean", "std", "median", "min", "max"]:
                        row[f"{metric}_{stat}"] = np.nan

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data).round(4)

        # Display as a formatted table
        self.logger.info("\nDistribution Summary by Outcome Variable")
        self.logger.info("=" * 80)

        # Show basic info first
        basic_cols = ["outcome_variable", "sample_size"] + [
            f"{m}_mean" for m in available_metrics
        ]
        basic_summary = summary_df[basic_cols]

        self.logger.info("\nMean Performance by Outcome:")
        self.logger.info(f"\n{basic_summary.to_string(index=False)}")

        return summary_df

    def plot_distribution_heatmap(
        self,
        metrics: List[str] = None,
        stat: str = "mean",
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Creates a heatmap of distribution statistics across outcomes and metrics.

        Args:
            metrics (Optional[List[str]], optional): A list of metrics to
                include. If None, uses a default list. Defaults to None.
            stat (str, optional): The statistic to display ('mean', 'std',
                'median', 'min', 'max'). Defaults to 'mean'.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (10, 6).

        Raises:
            ValueError: If 'outcome_variable' column is not found.
        """
        if metrics is None:
            metrics = ["auc", "mcc", "f1", "precision", "recall", "accuracy"]

        available_metrics = [col for col in metrics if col in self.clean_data.columns]

        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found")

        # Create summary data
        outcomes = sorted(self.clean_data["outcome_variable"].unique())
        if len(outcomes) > MAX_OUTCOMES_FOR_HEATMAP:
            warnings.warn(
                f"Found {len(outcomes)} outcomes. To keep the heatmap readable, displaying a random sample of {MAX_OUTCOMES_FOR_HEATMAP} outcomes. "
                "To plot specific outcomes, filter the input DataFrame before creating the plotter.",
                stacklevel=2,
            )
            import random

            outcomes = random.sample(outcomes, MAX_OUTCOMES_FOR_HEATMAP)

        heatmap_data = []

        for outcome in outcomes:
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]
            row = {"outcome": outcome}

            for metric in available_metrics:
                clean_values = outcome_data[metric].dropna()
                if len(clean_values) > 0:
                    if stat == "mean":
                        row[metric] = clean_values.mean()
                    elif stat == "std":
                        row[metric] = clean_values.std()
                    elif stat == "median":
                        row[metric] = clean_values.median()
                    elif stat == "min":
                        row[metric] = clean_values.min()
                    elif stat == "max":
                        row[metric] = clean_values.max()
                else:
                    row[metric] = np.nan

            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(heatmap_data).set_index("outcome")

        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": f"{stat.title()} Performance"},
        )

        plt.title(
            f"{stat.title()} Performance Heatmap by Outcome and Metric",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Outcome Variables", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


def plot_metric_correlation_by_outcome(
    data: pd.DataFrame,
    outcomes_to_plot: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
):
    """Plots correlation matrices of metrics, stratified by outcome variable.

    Args:
        data (pd.DataFrame): The results DataFrame.
        outcomes_to_plot (Optional[List[str]], optional): A list of specific
            outcomes to plot. If None, all are used. Defaults to None.
        figsize (Tuple[int, int], optional): The figure size.
            Defaults to (15, 10).
    Raises:
        ValueError: If 'outcome_variable' column is not found.
    """
    clean_data = get_clean_data(data)

    if "outcome_variable" not in clean_data.columns:
        raise ValueError("outcome_variable column not found")

    metrics = ["auc", "mcc", "f1", "precision", "recall", "accuracy"]
    available_metrics = [col for col in metrics if col in clean_data.columns]

    outcomes = outcomes_to_plot or sorted(clean_data["outcome_variable"].unique())
    if len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT:
        warnings.warn(
            f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
            f"Displaying the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
            "Use the 'outcomes_to_plot' parameter to select specific outcomes.",
            stacklevel=2,
        )
        outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

    n_outcomes = len(outcomes)

    # Calculate grid dimensions
    cols = min(3, n_outcomes)
    rows = (n_outcomes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n_outcomes == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        outcome_data = clean_data[clean_data["outcome_variable"] == outcome]

        if len(outcome_data) > 1:
            corr_matrix = outcome_data[available_metrics].corr()

            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )

            ax.set_title(
                f"{outcome}\n(n={len(outcome_data)})", fontsize=11, fontweight="bold"
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient Data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(f"{outcome}", fontsize=11)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Metric Correlations by Outcome Variable", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
