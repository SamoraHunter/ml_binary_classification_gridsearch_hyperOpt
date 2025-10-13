# plot_pipeline_parameters.py
"""
Pipeline parameter analysis plotting module for ML results analysis.
Focuses on visualizing the impact of data transformations and pipeline settings on model performance.
"""

import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional

from ml_grid.results_processing.core import get_clean_data

class PipelineParameterPlotter:
    """Visualizes the impact of pipeline parameters on model performance."""

    def __init__(self, data: pd.DataFrame):
        """Initializes the PipelineParameterPlotter.

        Args:
            data (pd.DataFrame): Results DataFrame, must contain columns for
                pipeline parameters and performance metrics.

        Raises:
            ValueError: If no pipeline parameter columns are found in the data.
        """
        self.data = data
        self.clean_data = get_clean_data(data)
        self.logger = logging.getLogger('ml_grid')
        
        # Define which parameters are categorical vs continuous based on the request
        self.categorical_params = [
            'resample', 'scale', 'param_space_size', 'percent_missing'
        ]
        # For continuous params, we map the column name to a user-friendly title
        self.continuous_params = {
            'nb_size': 'Number of Features Used', # Using nb_size for n_features as it represents selected features
            'X_train_size': 'Train Set Size',
            'X_test_orig_size': 'Original Test Set Size',
            'X_test_size': 'Test Set Size',
            'n_fits': 'Number of Fits (Random Search)',
            't_fits': 'Total Fits (Grid Search)'
        }

        # Filter to only parameters present in the data
        self.available_categorical = [p for p in self.categorical_params if p in self.clean_data.columns]
        self.available_continuous = {k: v for k, v in self.continuous_params.items() if k in self.clean_data.columns}

        if not self.available_categorical and not self.available_continuous:
            raise ValueError("No pipeline parameter columns (e.g., 'resample', 'scale') found in the provided data.")

        plt.style.use('default')
        sns.set_palette("viridis")

    def plot_categorical_parameters(
        self, metric: str = 'auc', figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """Creates box plots for categorical pipeline parameters.

        Args:
            metric (str, optional): The performance metric to plot.
                Defaults to 'auc'.
            figsize (Optional[Tuple[int, int]], optional): Figure size for the
                plot. Defaults to None.

        Raises:
            ValueError: If the specified metric is not found in the data.
        """
        if not self.available_categorical:
            self.logger.info("No categorical pipeline parameters found to plot.")
            return
            
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        n_params = len(self.available_categorical)
        cols = min(4, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig_size = figsize or (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, param in enumerate(self.available_categorical):
            ax = axes[i]
            
            # Handle potential NaN values by filling them with a string representation
            plot_data = self.clean_data.copy()
            if plot_data[param].isnull().any():
                plot_data[param] = plot_data[param].fillna('None')

            # For boolean 'scale', map to more descriptive labels
            if param == 'scale':
                 plot_data[param] = plot_data[param].astype(str).map({'True': 'Scaled', 'False': 'Not Scaled', 'None': 'None'})

            order = sorted(plot_data[param].unique())
            sns.boxplot(data=plot_data, x=param, y=metric, ax=ax, palette="Set3", order=order, hue=param, legend=False)
            ax.set_title(f'{param.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel(metric.upper() if i % cols == 0 else '')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.suptitle(f'Impact of Categorical Pipeline Parameters on {metric.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_continuous_parameters(
        self, metric: str = 'auc', figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """Creates scatter plots for continuous pipeline parameters.

        Args:
            metric (str, optional): The performance metric to plot.
                Defaults to 'auc'.
            figsize (Optional[Tuple[int, int]], optional): Figure size for the
                plot. Defaults to None.

        Raises:
            ValueError: If the specified metric is not found in the data.
        """
        if not self.available_continuous:
            self.logger.info("No continuous pipeline parameters found to plot.")
            return

        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        n_params = len(self.available_continuous)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig_size = figsize or (cols * 6, rows * 5)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, (param, title) in enumerate(self.available_continuous.items()):
            ax = axes[i]
            
            plot_data = self.clean_data.copy()
            # Ensure data is numeric and drop NaNs for plotting
            plot_data[param] = pd.to_numeric(plot_data[param], errors='coerce')
            plot_data.dropna(subset=[param, metric], inplace=True)
            
            if plot_data.empty:
                ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=11, fontweight='bold')
                continue

            sns.scatterplot(data=plot_data, x=param, y=metric, ax=ax, alpha=0.5, edgecolor=None, s=15)
            sns.regplot(data=plot_data, x=param, y=metric, ax=ax, scatter=False, color='red', line_kws={'linewidth': 2})
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel(title)
            ax.set_ylabel(metric.upper() if i % cols == 0 else '')
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.suptitle(f'Correlation of Continuous Pipeline Parameters with {metric.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()