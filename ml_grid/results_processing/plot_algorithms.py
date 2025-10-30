# plot_algorithms.py
"""
Algorithm comparison plotting module for ML results analysis.
Focuses on comparing algorithm performance with outcome variable stratification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from typing import List, Dict, Optional, Union, Tuple
from ml_grid.results_processing.core import get_clean_data
import warnings

# Maximum number of outcomes to display in stratified plots to avoid clutter.
MAX_OUTCOMES_FOR_STRATIFIED_PLOT = 20


class AlgorithmComparisonPlotter:
    """A class for creating algorithm comparison visualizations."""

    def __init__(self, data: pd.DataFrame):
        """Initializes the AlgorithmComparisonPlotter.

        Args:
            data (pd.DataFrame): A DataFrame containing the experiment results.
        """
        self.data = data
        self.clean_data = get_clean_data(data)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_algorithm_boxplots(
        self,
        metric: str = "auc",
        algorithms_to_plot: Optional[List[str]] = None,
        stratify_by_outcome: bool = False,
        outcomes_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """Creates box plots comparing algorithm performance.

        Args:
            metric (str, optional): The performance metric to compare.
                Defaults to 'auc'.
            algorithms_to_plot (Optional[List[str]], optional): A list of
                specific algorithms to include. If None, all are used.
                Defaults to None.
            stratify_by_outcome (bool, optional): If True, creates separate
                plots for each outcome. Defaults to False.
            outcomes_to_plot (Optional[List[str]], optional): A list of
                specific outcomes to plot. If None, all are used.
                Defaults to None.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (12, 6).

        Raises:
            ValueError: If the specified metric is not found in the data.
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        if not stratify_by_outcome:
            self._plot_single_algorithm_boxplot(metric, algorithms_to_plot, figsize)
        else:
            self._plot_stratified_algorithm_boxplots(
                metric, algorithms_to_plot, outcomes_to_plot, figsize
            )

    def _plot_single_algorithm_boxplot(
        self,
        metric: str,
        algorithms_to_plot: Optional[List[str]],
        figsize: Tuple[int, int],
    ) -> None:
        """Create single box plot for all outcomes combined."""
        plot_data = self.clean_data.copy()

        if algorithms_to_plot:
            plot_data = plot_data[plot_data["method_name"].isin(algorithms_to_plot)]

        plt.figure(figsize=figsize)

        # Create box plot
        sns.boxplot(
            data=plot_data, x="method_name", y=metric, showfliers=True, whis=1.5
        )

        # Add mean markers
        for i, algo in enumerate(plot_data["method_name"].unique()):
            algo_data = plot_data[plot_data["method_name"] == algo][metric]
            mean_val = algo_data.mean()
            plt.scatter(
                i,
                mean_val,
                color="red",
                s=100,
                marker="D",
                zorder=10,
                label="Mean" if i == 0 else "",
            )

        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Algorithm", fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(
            f"{metric.upper()} Performance by Algorithm - All Outcomes",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add sample size annotations
        for i, algo in enumerate(plot_data["method_name"].unique()):
            n_samples = len(plot_data[plot_data["method_name"] == algo])
            plt.text(
                i,
                plt.ylim()[0],
                f"n={n_samples}",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    def _plot_stratified_algorithm_boxplots(
        self,
        metric: str,
        algorithms_to_plot: Optional[List[str]],
        outcomes_to_plot: Optional[List[str]],
        figsize: Tuple[int, int],
    ) -> None:
        """Create stratified box plots by outcome variable."""
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

        # Calculate subplot layout
        cols = min(3, n_outcomes)
        rows = (n_outcomes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

        if n_outcomes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, outcome in enumerate(outcomes):
            ax = axes[i]
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]

            if algorithms_to_plot:
                outcome_data = outcome_data[
                    outcome_data["method_name"].isin(algorithms_to_plot)
                ]

            if len(outcome_data) > 0:
                sns.boxplot(data=outcome_data, x="method_name", y=metric, ax=ax)

                # Add means
                for j, algo in enumerate(outcome_data["method_name"].unique()):
                    algo_data = outcome_data[outcome_data["method_name"] == algo][
                        metric
                    ]
                    if len(algo_data) > 0:
                        mean_val = algo_data.mean()
                        ax.scatter(
                            j, mean_val, color="red", s=60, marker="D", zorder=10
                        )

                ax.tick_params(axis="x", rotation=45)
                plt.setp(ax.get_xticklabels(), ha="right")
                ax.set_title(
                    f"{outcome}\n{metric.upper()}", fontsize=11, fontweight="bold"
                )
                ax.set_xlabel("Algorithm" if i >= len(outcomes) - cols else "")
                ax.set_ylabel(metric.upper() if i % cols == 0 else "")
                ax.grid(True, alpha=0.3)
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
                ax.set_title(f"{outcome}", fontsize=11)

        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"{metric.upper()} Performance by Algorithm and Outcome",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_algorithm_performance_heatmap(
        self,
        metric: str = "auc",
        algorithms_to_plot: Optional[List[str]] = None,
        outcomes_to_plot: Optional[List[str]] = None,
        aggregation: str = "mean",
        figsize: Tuple[int, int] = (12, 8),
    ) -> pd.DataFrame:
        """Creates a heatmap showing algorithm performance across outcomes.

        Args:
            metric (str, optional): The performance metric to visualize.
                Defaults to 'auc'.
            algorithms_to_plot (Optional[List[str]], optional): A list of
                specific algorithms to include. Defaults to None.
            outcomes_to_plot (Optional[List[str]], optional): A list of
                specific outcomes to include. Defaults to None.
            aggregation (str, optional): How to aggregate multiple runs
                ('mean', 'median', 'max'). Defaults to 'mean'.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (12, 8).

        Raises:
            ValueError: If 'outcome_variable' column is missing or an invalid
                aggregation method is provided.

        Returns:
            pd.DataFrame: The pivot table data used for the heatmap.
        """
        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found")

        plot_data = self.clean_data.copy()

        if algorithms_to_plot:
            plot_data = plot_data[plot_data["method_name"].isin(algorithms_to_plot)]

        if outcomes_to_plot:
            plot_data = plot_data[plot_data["outcome_variable"].isin(outcomes_to_plot)]

        # Create pivot table
        if aggregation == "mean":
            heatmap_data = plot_data.pivot_table(
                values=metric,
                index="method_name",
                columns="outcome_variable",
                aggfunc="mean",
            )
        elif aggregation == "median":
            heatmap_data = plot_data.pivot_table(
                values=metric,
                index="method_name",
                columns="outcome_variable",
                aggfunc="median",
            )
        elif aggregation == "max":
            heatmap_data = plot_data.pivot_table(
                values=metric,
                index="method_name",
                columns="outcome_variable",
                aggfunc="max",
            )
        else:
            raise ValueError("aggregation must be 'mean', 'median', or 'max'")

        plt.figure(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": f"{aggregation.title()} {metric.upper()}"},
        )

        plt.title(
            f"{aggregation.title()} {metric.upper()} Performance Heatmap",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Outcome Variable", fontsize=12)
        plt.ylabel("Algorithm", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return heatmap_data

    def plot_algorithm_ranking(
        self,
        metric: str = "auc",
        algorithms_to_plot: Optional[List[str]] = None,
        stratify_by_outcome: bool = False,
        outcomes_to_plot: Optional[List[str]] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Plots a ranked bar chart of algorithm performance.

        Args:
            metric (str, optional): The performance metric to rank by.
                Defaults to 'auc'.
            algorithms_to_plot (Optional[List[str]], optional): A list of
                specific algorithms to include. Defaults to None.
            stratify_by_outcome (bool, optional): If True, creates separate
                plots for each outcome. Defaults to False.
            outcomes_to_plot (Optional[List[str]], optional): A list of
                specific outcomes to plot when stratified. Defaults to None.
            top_n (int, optional): The number of top algorithms to display.
                Defaults to 10.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (10, 8).

        Raises:
            ValueError: If the specified metric is not found, or if stratifying
                and 'outcome_variable' column is missing.
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        if not stratify_by_outcome:
            self._plot_single_ranking(metric, algorithms_to_plot, top_n, figsize)
        else:
            if "outcome_variable" not in self.clean_data.columns:
                raise ValueError("outcome_variable column not found for stratification")
            self._plot_stratified_ranking(
                metric, algorithms_to_plot, outcomes_to_plot, top_n, figsize
            )

    def _plot_single_ranking(
        self,
        metric: str,
        algorithms_to_plot: Optional[List[str]],
        top_n: int,
        figsize: Tuple[int, int],
    ) -> None:
        """Plot a single ranked bar chart for all outcomes combined."""
        plot_data = self.clean_data.copy()

        if algorithms_to_plot:
            plot_data = plot_data[plot_data["method_name"].isin(algorithms_to_plot)]

        # Calculate mean performance for each algorithm
        ranking = (
            plot_data.groupby("method_name")[metric].mean().sort_values(ascending=False)
        )

        # Select top N
        ranking = ranking.head(top_n)

        plt.figure(figsize=figsize)

        ax = sns.barplot(
            x=ranking.values,
            y=ranking.index,
            hue=ranking.index,
            orient="h",
            palette="viridis",
            legend=False,
        )

        ax.set_title(
            f"Top {top_n} Algorithms by Mean {metric.upper()} - All Outcomes",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"Mean {metric.upper()}", fontsize=12)
        ax.set_ylabel("Algorithm", fontsize=12)

        # Add value labels to bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        plt.tight_layout()
        plt.show()

    def _plot_stratified_ranking(
        self,
        metric: str,
        algorithms_to_plot: Optional[List[str]],
        outcomes_to_plot: Optional[List[str]],
        top_n: int,
        figsize: Tuple[int, int],
    ) -> None:
        """Plot stratified ranking bar charts by outcome."""
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

        cols = min(2, n_outcomes)
        rows = (n_outcomes + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 7, rows * 5), squeeze=False
        )
        axes = axes.flatten()

        for i, outcome in enumerate(outcomes):
            ax = axes[i]
            outcome_data = self.clean_data[
                self.clean_data["outcome_variable"] == outcome
            ]

            if algorithms_to_plot:
                outcome_data = outcome_data[
                    outcome_data["method_name"].isin(algorithms_to_plot)
                ]

            if len(outcome_data) > 0:
                ranking = (
                    outcome_data.groupby("method_name")[metric]
                    .mean()
                    .sort_values(ascending=False)
                    .head(top_n)
                )

                if not ranking.empty:
                    sns.barplot(
                        x=ranking.values,
                        y=ranking.index,
                        hue=ranking.index,
                        orient="h",
                        ax=ax,
                        palette="plasma",
                        legend=False,
                    )
                    ax.set_title(
                        f"{outcome} - Top {min(top_n, len(ranking))} Algorithms",
                        fontsize=11,
                        fontweight="bold",
                    )
                    ax.set_xlabel(f"Mean {metric.upper()}")
                    ax.set_ylabel("")

                    # Add value labels
                    for container in ax.containers:
                        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{outcome}", fontsize=11)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{outcome}", fontsize=11)

        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Top {top_n} Algorithms by Mean {metric.upper()} per Outcome",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_algorithm_stability(
        self, metric: str = "auc", top_n: int = 15, figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plots the stability (standard deviation) of algorithm performance.

        A lower standard deviation indicates more stable and predictable performance
        across different runs and data subsets.

        Args:
            metric (str, optional): The performance metric to evaluate
                stability on. Defaults to 'auc'.
            top_n (int, optional): The number of algorithms to display, ranked
                by stability (lower is better). Defaults to 15.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (10, 8).

        Raises:
            ValueError: If the specified metric is not found in the data.
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        # Calculate standard deviation for each algorithm
        stability = (
            self.clean_data.groupby("method_name")[metric]
            .std()
            .sort_values(ascending=True)
        )

        # Select top N most stable
        stability = stability.head(top_n)

        plt.figure(figsize=figsize)
        ax = sns.barplot(
            x=stability.values,
            y=stability.index,
            hue=stability.index,
            orient="h",
            palette="coolwarm_r",
            legend=False,
        )
        ax.set_title(
            f"Top {top_n} Most Stable Algorithms by {metric.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Standard Deviation of {metric.upper()} (Lower is Better)", fontsize=12
        )
        ax.set_ylabel("Algorithm", fontsize=12)

        ax.bar_label(ax.containers[0], fmt="%.4f", padding=3)
        plt.tight_layout()
        plt.show()

    def plot_performance_tradeoff(
        self,
        metric_y: str = "auc",
        metric_x: str = "run_time",
        stratify_by_outcome: bool = False,
        top_n_algos: Optional[int] = 10,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Plots a performance trade-off scatter plot between two metrics.

        This is useful for visualizing trade-offs like Performance vs. Speed.

        Args:
            metric_y (str, optional): The metric for the y-axis (e.g., 'auc').
                Defaults to 'auc'.
            metric_x (str, optional): The metric for the x-axis (e.g., 'run_time').
                Defaults to 'run_time'.
            stratify_by_outcome (bool, optional): If True, creates a separate
                plot for each outcome. Defaults to False.
            top_n_algos (Optional[int], optional): If set, only shows the top N
                algorithms based on `metric_y`. Defaults to 10.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (12, 8).

        Raises:
            ValueError: If one or both specified metrics are not found.
        """
        if (
            metric_y not in self.clean_data.columns
            or metric_x not in self.clean_data.columns
        ):
            raise ValueError(
                f"One or both metrics ('{metric_y}', '{metric_x}') not found in data."
            )

        plot_data = self.clean_data.copy()

        if top_n_algos:
            top_algos = (
                plot_data.groupby("method_name")[metric_y]
                .mean()
                .nlargest(top_n_algos)
                .index
            )
            plot_data = plot_data[plot_data["method_name"].isin(top_algos)]

        if not stratify_by_outcome:
            plt.figure(figsize=figsize)
            ax = sns.scatterplot(
                data=plot_data,
                x=metric_x,
                y=metric_y,
                hue="method_name",
                style="outcome_variable",
                alpha=0.7,
                s=80,
            )

            if (
                plot_data[metric_x].min() > 0
                and plot_data[metric_x].max() / plot_data[metric_x].min() > 100
            ):
                ax.set_xscale("log")
                plt.xlabel(
                    f'{metric_x.replace("_", " ").title()} (log scale)', fontsize=12
                )
            else:
                plt.xlabel(metric_x.replace("_", " ").title(), fontsize=12)

            plt.ylabel(metric_y.replace("_", " ").title(), fontsize=12)
            plt.title(
                f"Performance Trade-off: {metric_y.upper()} vs. {metric_x.upper()}",
                fontsize=14,
                fontweight="bold",
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Algorithm")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            # FacetGrid for stratified plots
            g = sns.FacetGrid(
                plot_data,
                col="outcome_variable",
                hue="method_name",
                col_wrap=min(3, plot_data["outcome_variable"].nunique()),
                sharex=False,
                sharey=True,
                height=5,
            )
            g.map(sns.scatterplot, metric_x, metric_y, alpha=0.8, s=60)
            g.add_legend(title="Algorithm")
            g.set_titles("{col_name}")
            g.set_axis_labels(
                metric_x.replace("_", " ").title(), metric_y.replace("_", " ").title()
            )
            g.fig.suptitle(
                f"Performance Trade-off by Outcome: {metric_y.upper()} vs. {metric_x.upper()}",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()
            plt.show()

    def plot_pareto_front(
        self,
        metric_y: str = "auc",
        metric_x: str = "run_time",
        lower_is_better_x: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Plots a Pareto front for two competing metrics.

        The Pareto front highlights the set of "optimal" algorithms where you cannot
        improve one metric without degrading the other.

        Args:
            metric_y (str, optional): The primary performance metric (higher is
                better). Defaults to 'auc'.
            metric_x (str, optional): The secondary metric, often a cost
                (e.g., 'run_time'). Defaults to 'run_time'.
            lower_is_better_x (bool, optional): Set to True if a lower value of
                `metric_x` is better. Defaults to True.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (12, 8).
        """
        # 1. Get mean performance for each algorithm
        summary_df = (
            self.clean_data.groupby("method_name")
            .agg(mean_y=(metric_y, "mean"), mean_x=(metric_x, "mean"))
            .reset_index()
        )

        # 2. Identify the Pareto front
        # A point is on the Pareto front if no other point dominates it.
        is_pareto = []
        for i, row in summary_df.iterrows():
            # Check if any other point dominates this one
            # Dominates = better on y AND better on x
            y_is_better = summary_df["mean_y"] > row["mean_y"]
            if lower_is_better_x:
                x_is_better = summary_df["mean_x"] < row["mean_x"]
            else:
                x_is_better = summary_df["mean_x"] > row["mean_x"]

            is_dominated = (y_is_better & x_is_better).any()
            is_pareto.append(not is_dominated)

        summary_df["is_pareto"] = is_pareto
        pareto_df = summary_df[summary_df["is_pareto"]].sort_values("mean_x")

        # 3. Plot
        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=summary_df,
            x="mean_x",
            y="mean_y",
            hue="is_pareto",
            style="is_pareto",
            s=100,
            palette={True: "red", False: "grey"},
            legend=False,
        )

        if not pareto_df.empty:
            plt.plot(pareto_df["mean_x"], pareto_df["mean_y"], "r--", alpha=0.7)

        # Annotate points
        for i, row in summary_df.iterrows():
            plt.text(
                row["mean_x"],
                row["mean_y"] * 1.001,
                row["method_name"],
                fontsize=9,
                ha="left",
                va="bottom",
            )

        plt.title(
            f"Pareto Front: {metric_y.upper()} vs {metric_x.title()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel(
            f'Mean {metric_x.title()}{" (Lower is Better)" if lower_is_better_x else ""}',
            fontsize=12,
        )
        plt.ylabel(f"Mean {metric_y.upper()} (Higher is Better)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_statistical_significance_heatmap(
        self,
        metric: str = "auc",
        outcome: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 12),
    ) -> None:
        """Performs pairwise t-tests and visualizes p-values in a heatmap.

        This helps determine if observed performance differences between
        algorithms are statistically significant.

        Args:
            metric (str, optional): The performance metric to compare.
                Defaults to 'auc'.
            outcome (Optional[str], optional): If specified, filters data for a
                single outcome. Otherwise, uses all data. Defaults to None.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (14, 12).

        Raises:
            ValueError: If stratifying and 'outcome_variable' column is missing.
        """
        plot_data = self.clean_data.copy()
        title = f"Pairwise T-test P-values for {metric.upper()}"
        if outcome:
            if "outcome_variable" not in plot_data.columns:
                raise ValueError(
                    "outcome_variable column not found for stratified analysis."
                )
            plot_data = plot_data[plot_data["outcome_variable"] == outcome]
            title += f" (Outcome: {outcome})"

        algorithms = sorted(plot_data["method_name"].unique())
        p_values = pd.DataFrame(np.nan, index=algorithms, columns=algorithms)

        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i <= j:
                    continue
                data1 = plot_data[plot_data["method_name"] == algo1][metric].dropna()
                data2 = plot_data[plot_data["method_name"] == algo2][metric].dropna()
                if len(data1) > 1 and len(data2) > 1:
                    _, p_val = ttest_ind(
                        data1, data2, equal_var=False, nan_policy="omit"
                    )
                    p_values.loc[algo1, algo2] = p_val
                    p_values.loc[algo2, algo1] = p_val

        plt.figure(figsize=figsize)
        sns.heatmap(
            p_values,
            annot=True,
            fmt=".3f",
            cmap="coolwarm_r",
            center=0.05,
            cbar_kws={"label": "P-value"},
        ).reset_index()
        plt.title(title, fontsize=14, fontweight="bold")
        plt.show()
