# plot_best_model.py
"""
Module for analyzing and visualizing the single best performing model for each outcome.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import warnings
import logging
import textwrap
import ast

from ml_grid.results_processing.core import get_clean_data
from ml_grid.results_processing.plot_hyperparameters import (
    HyperparameterAnalysisPlotter,
)  # To reuse parsing logic

# Limit on how many outcomes to plot automatically to avoid generating too many figures.
MAX_OUTCOMES_TO_PLOT = 10


class BestModelAnalyzerPlotter:
    """
    Analyzes and plots the characteristics of the best performing model for each outcome variable.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the plotter.

        Args:
            data: Aggregated results DataFrame. Must contain 'outcome_variable'.
        """
        if "outcome_variable" not in data.columns:
            raise ValueError(
                "Data must contain an 'outcome_variable' column for this analysis."
            )

        self.data = data
        self.clean_data = get_clean_data(data)
        self.logger = logging.getLogger("ml_grid")

        # Define feature categories and pipeline parameters from other modules for consistency
        self.feature_categories = [
            "age",
            "sex",
            "bmi",
            "ethnicity",
            "bloods",
            "diagnostic_order",
            "drug_order",
            "annotation_n",
            "meta_sp_annotation_n",
            "meta_sp_annotation_mrc_n",
            "annotation_mrc_n",
            "core_02",
            "bed",
            "vte_status",
            "hosp_site",
            "core_resus",
            "news",
            "date_time_stamp",
        ]
        self.pipeline_params = [
            "resample",
            "scale",
            "param_space_size",
            "percent_missing",
        ]

        plt.style.use("default")
        sns.set_palette("muted")

    def _get_best_models(self, metric: str) -> pd.DataFrame:
        """Finds the single best model for each outcome variable.

        Args:
            metric (str): The performance metric to use for determining the best model.

        Returns:
            pd.DataFrame: A DataFrame containing the single best run for each
            outcome, sorted by the specified metric in descending order.
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in the data.")

        # Find the index of the maximum metric value for each outcome group
        best_indices = self.clean_data.loc[
            self.clean_data.groupby("outcome_variable")[metric].idxmax()
        ]

        return best_indices.sort_values(by=metric, ascending=False)

    def plot_best_model_summary(
        self,
        metric: str = "auc",
        outcomes_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 9),
    ):
        """Generates a summary plot for the best model of each outcome.

        This method finds the best performing model for each outcome and creates
        a detailed 2x2 plot summarizing its algorithm, performance,
        hyperparameters, and pipeline settings.

        Args:
            metric (str, optional): The metric to determine the "best" model.
                Defaults to 'auc'.
            outcomes_to_plot (Optional[List[str]], optional): A specific list of
                outcomes to analyze. If None, analyzes all outcomes up to a limit.
                Defaults to None.
            figsize (Tuple[int, int], optional): The figure size for each summary
                plot. Defaults to (14, 9).
        """
        best_models_df = self._get_best_models(metric)

        if outcomes_to_plot:
            # Filter to only the requested outcomes
            best_models_df = best_models_df[
                best_models_df["outcome_variable"].isin(outcomes_to_plot)
            ]
            if best_models_df.empty:
                self.logger.warning(
                    f"No data found for the specified outcomes: {outcomes_to_plot}"
                )
                return
        elif len(best_models_df) > MAX_OUTCOMES_TO_PLOT:
            warnings.warn(
                f"Found {len(best_models_df)} unique outcomes. To avoid excessive plotting, "
                f"showing summaries for the top {MAX_OUTCOMES_TO_PLOT} outcomes based on the '{metric}' metric. "
                "Use the 'outcomes_to_plot' parameter to specify which outcomes to analyze.",
                stacklevel=2,
            )
            best_models_df = best_models_df.head(MAX_OUTCOMES_TO_PLOT)

        self.logger.info(
            f"--- Generating Best Model Summaries (Metric: {metric.upper()}) ---"
        )
        for _, model_series in best_models_df.iterrows():
            self._plot_single_model_summary(model_series, metric, figsize)

    def _plot_single_model_summary(
        self, model_series: pd.Series, metric: str, figsize: Tuple[int, int]
    ):
        """Generates a single 2x2 summary plot for one model series.

        Args:
            model_series (pd.Series): A row from the DataFrame representing the
                best model for a single outcome.
            metric (str): The primary performance metric being used.
            figsize (Tuple[int, int]): The figure size for the plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Best Model Analysis for: {model_series['outcome_variable']}",
            fontsize=16,
            fontweight="bold",
        )

        # Subplot 1: Key Information (Text)
        self._plot_key_info(axes[0, 0], model_series, metric)

        # Subplot 2: Hyperparameters
        self._plot_hyperparameters(axes[0, 1], model_series)

        # Subplot 3: Feature Categories Used
        self._plot_feature_categories(axes[1, 0], model_series)

        # Subplot 4: Pipeline Parameters
        self._plot_pipeline_parameters(axes[1, 1], model_series)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_key_info(self, ax: plt.Axes, model_series: pd.Series, metric: str):
        """Plots key model and performance info on a given axis.

        Args:
            ax (plt.Axes): The matplotlib axis to plot on.
            model_series (pd.Series): The data for the best model.
            metric (str): The name of the primary metric.
        """
        ax.set_title("Model & Performance Summary", fontsize=12, fontweight="bold")
        ax.axis("off")

        score = model_series.get(metric, "N/A")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

        info_text = (
            f"Algorithm: {model_series.get('method_name', 'N/A')}\n"
            f"Best Score ({metric.upper()}): {score_str}\n"
            f"Number of Features: {model_series.get('nb_size', 'N/A')}\n"
            f"Run Timestamp: {model_series.get('run_timestamp', 'N/A')}\n\n"
            f"Other Metrics:\n"
            f"  - F1: {model_series.get('f1', 'N/A'):.4f}\n"
            f"  - MCC: {model_series.get('mcc', 'N/A'):.4f}\n"
            f"  - Accuracy: {model_series.get('accuracy', 'N/A'):.4f}\n"
        )

        ax.text(
            0.05,
            0.95,
            info_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="grey", lw=1),
        )

    def _plot_hyperparameters(self, ax: plt.Axes, model_series: pd.Series):
        """Plots the hyperparameters of the model on a given axis.

        Args:
            ax (plt.Axes): The matplotlib axis to plot on.
            model_series (pd.Series): The data for the best model.
        """
        ax.set_title("Hyperparameters", fontsize=12, fontweight="bold")
        ax.axis("off")

        params = {}
        if "algorithm_implementation" in model_series and pd.notna(
            model_series["algorithm_implementation"]
        ):
            # Reuse parsing logic from HyperparameterAnalysisPlotter
            params = HyperparameterAnalysisPlotter._parse_model_string_to_params(
                model_series["algorithm_implementation"]
            )

        if not params:
            ax.text(
                0.5,
                0.5,
                "Hyperparameters not available\nor could not be parsed.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            return

        param_str = ""
        for key, val in params.items():
            val_str = str(val)
            if len(val_str) > 40:
                val_str = textwrap.fill(val_str, width=40, subsequent_indent="    ")
            param_str += f"{key}: {val_str}\n"

        ax.text(
            0.05,
            0.95,
            param_str.strip(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="grey", lw=1),
        )

    def _plot_feature_categories(self, ax: plt.Axes, model_series: pd.Series):
        """Plots which feature categories were used on a given axis.

        Args:
            ax (plt.Axes): The matplotlib axis to plot on.
            model_series (pd.Series): The data for the best model.
        """
        ax.set_title("Feature Categories Used", fontsize=12, fontweight="bold")

        used_categories = {}
        for cat in self.feature_categories:
            if cat in model_series and pd.notna(model_series[cat]):
                val = model_series[cat]
                try:
                    is_used = (
                        ast.literal_eval(str(val).capitalize())
                        if isinstance(val, str)
                        else bool(val)
                    )
                except (ValueError, SyntaxError):
                    is_used = False

                if is_used:
                    used_categories[cat.replace("_", " ").title()] = 1

        if not used_categories:
            ax.text(
                0.5,
                0.5,
                "No feature category information available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        cat_df = pd.DataFrame.from_dict(
            used_categories, orient="index", columns=["Used"]
        ).sort_index()

        sns.barplot(
            x=cat_df.index,
            y=cat_df["Used"],
            ax=ax,
            palette="viridis",
            hue=cat_df.index,
            legend=False,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("Enabled")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["", "Yes"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    def _plot_pipeline_parameters(self, ax: plt.Axes, model_series: pd.Series):
        """Plots the pipeline settings on a given axis.

        Args:
            ax (plt.Axes): The matplotlib axis to plot on.
            model_series (pd.Series): The data for the best model.
        """
        ax.set_title("Pipeline Settings", fontsize=12, fontweight="bold")
        ax.axis("off")

        pipeline_settings = {}
        for param in self.pipeline_params:
            if param in model_series and pd.notna(model_series[param]):
                pipeline_settings[param.replace("_", " ").title()] = model_series[param]

        if not pipeline_settings:
            ax.text(
                0.5,
                0.5,
                "No pipeline setting information available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            return

        settings_str = ""
        for key, val in pipeline_settings.items():
            settings_str += f"{key}: {val}\n"

        ax.text(
            0.05,
            0.95,
            settings_str.strip(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="honeydew", ec="grey", lw=1),
        )
