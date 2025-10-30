# plot_feature_categories.py
"""
Feature category analysis plotting module for ML results analysis.
Focuses on visualizing the impact of including different data source categories on model performance.
"""

import ast
import logging
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml_grid.results_processing.core import get_clean_data


class FeatureCategoryPlotter:
    """Visualizes the impact of feature categories on model performance.

    These categories correspond to the boolean flags that control which data
    sources are included at the start of the data pipeline.
    """

    def __init__(self, data: pd.DataFrame):
        """Initializes the FeatureCategoryPlotter.

        Args:
            data (pd.DataFrame): Results DataFrame, must contain boolean columns
                for feature categories and performance metrics.
        Raises:
            ValueError: If no feature category columns are found in the data.
        """
        self.data = data
        self.clean_data = get_clean_data(data)
        self.logger = logging.getLogger("ml_grid")
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
        # Filter to only categories present in the data
        self.available_categories = [
            cat for cat in self.feature_categories if cat in self.clean_data.columns
        ]

        if not self.available_categories:
            raise ValueError(
                "No feature category columns (e.g., 'bloods', 'age') found in the provided data."
            )

        plt.style.use("default")
        sns.set_palette("viridis")

    def plot_category_performance_boxplots(
        self, metric: str = "auc", figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """Creates box plots comparing performance when a feature category is included.

        Args:
            metric (str, optional): The performance metric to plot.
                Defaults to 'auc'.
            figsize (Optional[Tuple[int, int]], optional): Figure size for the
                plot. Defaults to None.
        Raises:
            ValueError: If the specified metric is not found in the data.
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        n_categories = len(self.available_categories)
        cols = min(4, n_categories)
        rows = (n_categories + cols - 1) // cols

        fig_size = figsize or (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, category in enumerate(self.available_categories):
            ax = axes[i]

            # Make a copy to avoid SettingWithCopyWarning
            plot_data = self.clean_data.copy()

            # Ensure the category column is boolean or can be treated as such
            if plot_data[category].dtype != bool:
                try:
                    # ast.literal_eval is safer for 'True'/'False' strings
                    if plot_data[category].apply(type).eq(str).all():
                        plot_data[category] = plot_data[category].apply(
                            ast.literal_eval
                        )
                    plot_data[category] = plot_data[category].astype(bool)
                except Exception:
                    warnings.warn(
                        f"Could not convert column '{category}' to boolean. Skipping.",
                        stacklevel=2,
                    )
                    ax.text(
                        0.5,
                        0.5,
                        "Invalid Data Type",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(category, fontsize=11)
                    continue

            sns.boxplot(
                data=plot_data,
                x=category,
                y=metric,
                ax=ax,
                palette="Set2",
                hue=category,
                legend=False,
            )
            ax.set_title(
                f'{category.replace("_", " ").title()}', fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Included in Pipeline")
            ax.set_ylabel(metric.upper() if i % cols == 0 else "")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Impact of Including Feature Categories on {metric.upper()}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_category_impact_on_metric(
        self, metric: str = "auc", figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plots the impact of including each feature category on a metric.

        Impact is calculated as:
        (Mean metric with category) - (Mean metric without category)

        Args:
            metric (str, optional): The performance metric to evaluate.
                Defaults to 'auc'.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (10, 8).
        Raises:
            ValueError: If the specified metric is not found in the data.
        """
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        plot_data = self.clean_data.copy()
        impact_data = []
        for category in self.available_categories:
            # Ensure boolean type
            if plot_data[category].dtype != bool:
                try:
                    if plot_data[category].apply(type).eq(str).all():
                        plot_data[category] = plot_data[category].apply(
                            ast.literal_eval
                        )
                    plot_data[category] = plot_data[category].astype(bool)
                except Exception:
                    continue

            # Check if both True and False values exist
            if plot_data[category].nunique() < 2:
                continue

            mean_with = plot_data[plot_data[category] == True][metric].mean()
            mean_without = plot_data[plot_data[category] == False][metric].mean()

            impact = mean_with - mean_without

            if not pd.isna(impact):
                impact_data.append(
                    {
                        "category": category.replace("_", " ").title(),
                        "impact": impact,
                    }
                )

        if not impact_data:
            self.logger.info(
                "Could not calculate impact for any feature categories. This may be because no categories had both included and excluded runs."
            )
            return

        impact_df = pd.DataFrame(impact_data).sort_values("impact", ascending=False)

        plt.figure(figsize=figsize)

        colors = ["#3a923a" if x > 0 else "#c14242" for x in impact_df["impact"]]
        ax = sns.barplot(
            x="impact",
            y="category",
            data=impact_df,
            orient="h",
            palette=colors,
            hue="category",
            legend=False,
        )

        ax.set_title(
            f"Impact of Including Feature Category on {metric.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Change in Mean {metric.upper()} (Included vs. Excluded)", fontsize=12
        )
        ax.set_ylabel("Feature Category", fontsize=12)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)

        plt.tight_layout()
        plt.show()
