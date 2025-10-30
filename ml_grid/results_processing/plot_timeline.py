# plot_timeline.py
"""
Timeline analysis plotting module for ML results analysis.
Focuses on temporal trends and run-to-run comparisons with outcome stratification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
from ml_grid.results_processing.core import get_clean_data
import logging
import warnings

# Maximum number of outcomes to display in stratified plots to avoid clutter.
MAX_OUTCOMES_FOR_STRATIFIED_PLOT = 8


class TimelineAnalysisPlotter:
    """A class for creating timeline and temporal analysis visualizations."""

    def __init__(self, data: pd.DataFrame):
        """Initializes the timeline analysis plotter.

        Args:
            data (pd.DataFrame): Results DataFrame, which must contain a
                'run_timestamp' column.

        Raises:
            ValueError: If the 'run_timestamp' column is not found in the data.
        """
        self.data = data
        self.clean_data = get_clean_data(data)
        self.logger = logging.getLogger("ml_grid")

        if "run_timestamp" not in self.data.columns:
            raise ValueError("run_timestamp column required for timeline analysis")

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Sort data by timestamp
        self.clean_data = self.clean_data.sort_values("run_timestamp")

    def plot_performance_timeline(
        self,
        metric: str = "auc",
        algorithms_to_plot: Optional[List[str]] = None,
        stratify_by_outcome: bool = False,
        outcomes_to_plot: Optional[List[str]] = None,
        aggregation: str = "mean",
        figsize: Tuple[int, int] = (14, 6),
    ) -> None:
        """Plots performance metrics over time (across runs).

        Args:
            metric (str, optional): The performance metric to plot.
                Defaults to 'auc'.
            algorithms_to_plot (Optional[List[str]], optional): A list of
                specific algorithms to include. Defaults to None.
            stratify_by_outcome (bool, optional): If True, creates separate
                plots for each outcome. Defaults to False.
            outcomes_to_plot (Optional[List[str]], optional): A list of specific
                outcomes to plot. Defaults to None.
            aggregation (str, optional): How to aggregate within runs ('mean',
                'best', 'median'). Defaults to 'mean'.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (14, 6).
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        if not stratify_by_outcome:
            self._plot_single_timeline(metric, algorithms_to_plot, aggregation, figsize)
        else:
            self._plot_stratified_timeline(
                metric, algorithms_to_plot, outcomes_to_plot, aggregation, figsize
            )

    def _plot_single_timeline(
        self,
        metric: str,
        algorithms_to_plot: Optional[List[str]],
        aggregation: str,
        figsize: Tuple[int, int],
    ) -> None:
        """Helper to plot a single timeline for all outcomes combined."""
        plot_data = self.clean_data.copy()

        if algorithms_to_plot:
            plot_data = plot_data[plot_data["method_name"].isin(algorithms_to_plot)]

        # Aggregate by run and algorithm
        if aggregation == "mean":
            timeline_data = plot_data.groupby(["run_timestamp", "method_name"])[
                metric
            ].mean()
        elif aggregation == "best":
            timeline_data = plot_data.groupby(["run_timestamp", "method_name"])[
                metric
            ].max()
        elif aggregation == "median":
            timeline_data = plot_data.groupby(["run_timestamp", "method_name"])[
                metric
            ].median()
        else:
            raise ValueError("aggregation must be 'mean', 'best', or 'median'")

        timeline_df = timeline_data.unstack(fill_value=np.nan)

        plt.figure(figsize=figsize)

        # Convert index to datetime
        timeline_df.index = pd.to_datetime(timeline_df.index, errors="coerce")

        # Plot each algorithm
        for algo in timeline_df.columns:
            algo_data = timeline_df[algo].dropna()
            if len(algo_data) > 0:
                plt.plot(
                    algo_data.index,
                    algo_data,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=algo,
                )

        plt.xlabel("Run Timestamp", fontsize=12)
        plt.ylabel(f"{aggregation.title()} {metric.upper()}", fontsize=12)
        plt.title(
            f"{aggregation.title()} {metric.upper()} Performance Timeline - All Outcomes",
            fontsize=14,
            fontweight="bold",
        )

        plt.xticks(rotation=45, ha="right")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_stratified_timeline(
        self,
        metric: str,
        algorithms_to_plot: Optional[List[str]],
        outcomes_to_plot: Optional[List[str]],
        aggregation: str,
        figsize: Tuple[int, int],
    ) -> None:
        """Helper to plot timelines stratified by outcome variable."""
        if "outcome_variable" not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found for stratification")

        outcomes = outcomes_to_plot or sorted(
            self.clean_data["outcome_variable"].unique()
        )
        if (
            len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT
            and outcomes_to_plot is None
        ):
            warnings.warn(
                f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Displaying the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                "Use the 'outcomes_to_plot' parameter to select specific outcomes.",
                stacklevel=2,
            )
            outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        n_outcomes = len(outcomes)

        fig, axes = plt.subplots(
            n_outcomes, 1, figsize=(figsize[0], figsize[1] * n_outcomes / 2)
        )

        if n_outcomes == 1:
            axes = [axes]

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
                # Aggregate by run and algorithm
                if aggregation == "mean":
                    timeline_data = outcome_data.groupby(
                        ["run_timestamp", "method_name"]
                    )[metric].mean()
                elif aggregation == "best":
                    timeline_data = outcome_data.groupby(
                        ["run_timestamp", "method_name"]
                    )[metric].max()
                elif aggregation == "median":
                    timeline_data = outcome_data.groupby(
                        ["run_timestamp", "method_name"]
                    )[metric].median()

                timeline_df = timeline_data.unstack(fill_value=np.nan)

                # Plot each algorithm
                for algo in timeline_df.columns:
                    algo_data = timeline_df[algo].dropna()
                    if len(algo_data) > 0:
                        x_positions = range(len(algo_data))
                        ax.plot(
                            x_positions,
                            algo_data,
                            marker="o",
                            linewidth=2,
                            markersize=4,
                            label=algo,
                        )

                ax.set_title(
                    f"{outcome} - {aggregation.title()} {metric.upper()}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_ylabel(f"{metric.upper()}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
                ax.grid(True, alpha=0.3)

                # Set x-tick labels
                if i == len(outcomes) - 1:  # Only on bottom plot
                    run_timestamps = timeline_df.index.tolist()
                    tick_positions = range(
                        0, len(run_timestamps), max(1, len(run_timestamps) // 8)
                    )
                    tick_labels = [
                        run_timestamps[j][:10] + "..." for j in tick_positions
                    ]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
                    ax.set_xlabel("Run Index (Chronological Order)")
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
                ax.set_title(f"{outcome}", fontsize=12)

        plt.suptitle(
            f"{aggregation.title()} {metric.upper()} Timeline by Outcome",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_improvement_trends(
        self,
        metric: str = "auc",
        algorithms_to_plot: Optional[List[str]] = None,
        stratify_by_outcome: bool = False,
        outcomes_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 7),
    ) -> None:
        """Plots the optimization progress within each run.

        This helps visualize how quickly the optimization finds better models within each batch/run.

        Args:
            metric (str, optional): The performance metric to analyze.
                Defaults to 'auc'.
            algorithms_to_plot (Optional[List[str]], optional): A list of
                specific algorithms to include. If None, all are used.
                Defaults to None.
            stratify_by_outcome (bool, optional): If True, creates separate
                plots for each outcome. Defaults to False.
            outcomes_to_plot (Optional[List[str]], optional): A list of specific
                outcomes to plot if stratified. Defaults to None.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (14, 7).
        """
        plot_data = self.clean_data.copy()

        if algorithms_to_plot:
            plot_data = plot_data[plot_data["method_name"].isin(algorithms_to_plot)]

        if not stratify_by_outcome:
            self._plot_single_intra_run_progress(plot_data, metric, figsize)
        else:
            self._plot_stratified_intra_run_progress(
                plot_data, metric, outcomes_to_plot, figsize
            )

    def _plot_single_intra_run_progress(
        self, plot_data: pd.DataFrame, metric: str, figsize: Tuple[int, int]
    ) -> None:
        """Helper to plot optimization progress for multiple runs on a single plot."""
        plt.figure(figsize=figsize)

        runs = plot_data["run_timestamp"].unique()

        for run_id in runs:
            run_data = plot_data[plot_data["run_timestamp"] == run_id]

            if run_data.empty:
                continue

            # The order of rows within each run_timestamp group is preserved from the original CSV,
            # which represents the trial order.
            # .reset_index(drop=True) is important to get a simple 0-based index for plotting trials
            cummax_perf = run_data[metric].cummax().reset_index(drop=True)

            # Plot trials vs best-so-far
            plt.plot(
                cummax_perf.index, cummax_perf, label=f"Run {run_id[:10]}...", alpha=0.8
            )

        plt.xlabel("Trial Number (within run)", fontsize=12)
        plt.ylabel(f"Best {metric.upper()} So Far", fontsize=12)
        plt.title(
            f"Optimization Progress per Run - All Outcomes",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_stratified_intra_run_progress(
        self,
        plot_data: pd.DataFrame,
        metric: str,
        outcomes_to_plot: Optional[List[str]],
        figsize: Tuple[int, int],
    ) -> None:
        """Helper to plot optimization progress stratified by outcome."""
        if "outcome_variable" not in plot_data.columns:
            raise ValueError("outcome_variable column not found for stratification")

        outcomes = outcomes_to_plot or sorted(plot_data["outcome_variable"].unique())
        if (
            len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT
            and outcomes_to_plot is None
        ):
            warnings.warn(
                f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Displaying the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                "Use the 'outcomes_to_plot' parameter to select specific outcomes.",
                stacklevel=2,
            )
            outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        n_outcomes = len(outcomes)

        if n_outcomes == 0:
            self.logger.info("No outcomes to plot for stratified improvement trends.")
            return

        fig, axes = plt.subplots(
            n_outcomes,
            1,
            figsize=(figsize[0], figsize[1] * n_outcomes / 2.5),
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()

        for i, outcome in enumerate(outcomes):
            ax = axes[i]
            outcome_data = plot_data[plot_data["outcome_variable"] == outcome]

            if outcome_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{outcome}", fontsize=12)
                continue

            runs = outcome_data["run_timestamp"].unique()

            for run_id in runs:
                run_outcome_data = outcome_data[outcome_data["run_timestamp"] == run_id]

                if run_outcome_data.empty:
                    continue

                cummax_perf = run_outcome_data[metric].cummax().reset_index(drop=True)
                ax.plot(
                    cummax_perf.index,
                    cummax_perf,
                    label=f"Run {run_id[:10]}...",
                    alpha=0.7,
                )

            ax.set_title(f"{outcome}", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"Best {metric.upper()} So Far")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

        if n_outcomes > 0:
            # Set xlabel only on the bottom-most plot that is visible
            for ax in reversed(axes):
                if ax.get_visible():
                    ax.set_xlabel("Trial Number (within run)", fontsize=12)
                    break

        plt.suptitle(
            f"Optimization Progress per Run by Outcome", fontsize=16, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def plot_computational_cost_timeline(
        self,
        algorithms_to_plot: Optional[List[str]] = None,
        stratify_by_outcome: bool = False,
        outcomes_to_plot: Optional[List[str]] = None,
        aggregation: str = "mean",
        figsize: Tuple[int, int] = (14, 6),
    ) -> None:
        """Plots the computational cost (run_time) over time (across runs).

        Args:
            algorithms_to_plot (Optional[List[str]], optional): A list of
                specific algorithms to include. Defaults to None.
            stratify_by_outcome (bool, optional): If True, creates separate
                plots for each outcome. Defaults to False.
            outcomes_to_plot (Optional[List[str]], optional): A list of specific
                outcomes to plot if stratified. Defaults to None.
            aggregation (str, optional): How to aggregate within runs ('mean',
                'sum', 'median'). Defaults to 'mean'.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (14, 6).
        """
        if "run_time" not in self.clean_data.columns:
            warnings.warn(
                "'run_time' column not found. Skipping computational cost plot.",
                stacklevel=2,
            )
            return

        self.logger.info("\nGenerating Computational Cost Timeline (run_time)...")
        # Re-use the existing timeline plotting logic for the 'run_time' metric
        self.plot_performance_timeline(
            metric="run_time",
            algorithms_to_plot=algorithms_to_plot,
            stratify_by_outcome=stratify_by_outcome,
            outcomes_to_plot=outcomes_to_plot,
            aggregation=aggregation,
            figsize=figsize,
        )
