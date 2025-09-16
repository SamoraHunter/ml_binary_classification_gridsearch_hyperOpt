# plot_interactions.py
"""
Interaction effect plotting module for ML results analysis.
Focuses on visualizing how pairs of parameters jointly affect model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

from ml_grid.results_processing.core import get_clean_data

class InteractionPlotter:
    """Visualizes interaction effects between experimental parameters.

    This class helps to understand how pairs of parameters (e.g., pipeline
    settings, feature categories) jointly affect model performance.
    """

    def __init__(self, data: pd.DataFrame):
        """Initializes the InteractionPlotter.

        Args:
            data (pd.DataFrame): Results DataFrame, must contain columns for
                parameters and performance metrics.
        """
        self.data = data
        self.clean_data = get_clean_data(data)
        plt.style.use('default')

    def plot_categorical_interaction(
        self,
        param1: str,
        param2: str,
        metric: str = 'auc',
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Visualizes the interaction of two categorical parameters via a heatmap.

        Args:
            param1 (str): The name of the first categorical parameter column.
            param2 (str): The name of the second categorical parameter column.
            metric (str, optional): The performance metric to plot.
                Defaults to 'auc'.
            figsize (Tuple[int, int], optional): Figure size for the plot.
                Defaults to (10, 8).
        """
        if param1 not in self.clean_data.columns or param2 not in self.clean_data.columns:
            print(f"Warning: One or both parameters ('{param1}', '{param2}') not found. Skipping.")
            return
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        plot_data = self.clean_data.copy()

        # Handle potential NaN values by filling them with a string representation
        for param in [param1, param2]:
            if plot_data[param].isnull().any():
                plot_data[param] = plot_data[param].fillna('None')

        try:
            pivot_table = plot_data.pivot_table(
                values=metric, index=param1, columns=param2, aggfunc='mean'
            )
        except Exception as e:
            print(f"Could not create pivot table for '{param1}' and '{param2}'. Reason: {e}")
            return

        plt.figure(figsize=figsize)
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
        plt.title(
            f'Interaction of {param1.replace("_", " ").title()} and {param2.replace("_", " ").title()} on {metric.upper()}',
            fontsize=14,
            fontweight='bold',
        )
        plt.xlabel(param2.replace("_", " ").title(), fontsize=12)
        plt.ylabel(param1.replace("_", " ").title(), fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_continuous_interaction(
        self,
        param1: str,
        param2: str,
        metric: str = 'auc',
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """Visualizes the interaction of two continuous parameters.

        Uses a scatter plot where point color represents the metric value.

        Args:
            param1 (str): The name of the first continuous parameter column.
            param2 (str): The name of the second continuous parameter column.
            metric (str, optional): The performance metric to use for color.
                Defaults to 'auc'.
            figsize (Tuple[int, int], optional): Figure size for the plot.
                Defaults to (10, 8).
        """
        if param1 not in self.clean_data.columns or param2 not in self.clean_data.columns:
            print(f"Warning: One or both parameters ('{param1}', '{param2}') not found. Skipping.")
            return
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        plot_data = self.clean_data[[param1, param2, metric]].dropna()

        if plot_data.empty:
            print(f"No data to plot for interaction between '{param1}' and '{param2}'.")
            return

        plt.figure(figsize=figsize)
        scatter = plt.scatter(plot_data[param1], plot_data[param2], c=plot_data[metric], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=metric.upper())
        plt.title(f'Interaction of {param1.title()} and {param2.title()} on {metric.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel(param1.replace("_", " ").title(), fontsize=12)
        plt.ylabel(param2.replace("_", " ").title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
