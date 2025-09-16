"""
Hyperparameter analysis plotting module for ML results analysis.
Focuses on visualizing the impact of hyperparameters on model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import warnings
import ast
import scipy.stats as stats
from sklearn.metrics import r2_score

from ml_grid.results_processing.core import get_clean_data

class HyperparameterAnalysisPlotter:
    """Analyzes and visualizes the impact of hyperparameters on model performance.

    This class extracts hyperparameter settings from model string representations
    in the results data, allowing for detailed analysis of how different
    hyperparameters affect a given performance metric.
    """
    # Define algorithms to ignore for hyperparameter parsing as they don't store them in a parsable format.
    _ALGOS_TO_IGNORE = ['CatBoostClassifier', 'KNNWrapper', 'knn_wrapper_class']

    def __init__(self, data: pd.DataFrame):
        if 'algorithm_implementation' not in data.columns:
            raise ValueError("Data must contain an 'algorithm_implementation' column for hyperparameter analysis.")
        
        self.data = data
        self.clean_data = get_clean_data(data)
        
        # Extract algorithm name from algorithm_implementation
        self.clean_data['algorithm_name'] = self.clean_data['algorithm_implementation'].apply(
            lambda x: x.split('(')[0].strip() if isinstance(x, str) and '(' in x else None
        )

        # Filter out ignored algorithms before parsing
        self.clean_data = self.clean_data[~self.clean_data['algorithm_name'].isin(self._ALGOS_TO_IGNORE)]
        
        # Parse parameters
        self.clean_data['params_dict'] = self.clean_data['algorithm_implementation'].apply(
            self._parse_model_string_to_params
        )
        
        # Drop rows where parsing failed
        self.clean_data = self.clean_data.dropna(subset=['params_dict', 'algorithm_name']).copy()

        plt.style.use('default')
        sns.set_palette("muted")

    @staticmethod
    def _parse_model_string_to_params(model_str: str) -> Optional[Dict[str, Any]]:
        """Parses a scikit-learn model's string representation into a parameter dictionary.

        This method uses Abstract Syntax Trees (AST) to safely parse the
        string representation of a model (e.g., "RandomForestClassifier(n_estimators=100)")
        and extract its hyperparameters into a dictionary.

        Args:
            model_str (str): The string representation of the model.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of the model's hyperparameters,
            or None if parsing fails.
        """
        if not isinstance(model_str, str) or '(' not in model_str:
            return None
        try:
            # Handle sklearn-style string representations
            tree = ast.parse(model_str, mode='eval')
            
            if not isinstance(tree.body, ast.Call):
                return None
                
            params = {}
            for kw in tree.body.keywords:
                try:
                    params[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    if isinstance(kw.value, ast.Name):
                        val_id = kw.value.id
                        if val_id == 'True':
                            params[kw.arg] = True
                        elif val_id == 'False':
                            params[kw.arg] = False
                        elif val_id == 'None':
                            params[kw.arg] = None
                        else:
                            params[kw.arg] = val_id
                    elif isinstance(kw.value, ast.Constant):
                        params[kw.arg] = kw.value.value
                    else:
                        # Store as string representation
                        params[kw.arg] = ast.unparse(kw.value)
            return params
        except Exception as e:
            warnings.warn(f"Failed to parse model string '{model_str[:50]}...'. Error: {e}")
            return None

    def get_available_algorithms(self):
        """Gets a list of available, parsable algorithms from the data.

        Returns:
            List[str]: A sorted list of unique algorithm names.
        """
        return sorted(self.clean_data['algorithm_name'].unique())

    def plot_performance_by_hyperparameter(self,
                                           algorithm_name: str,
                                           hyperparameters: List[str],
                                           metric: str = 'auc',
                                           figsize: Optional[Tuple[int, int]] = None):
        """Plots performance against a list of hyperparameters in a grid.

        This function provides a visual analysis of how individual parameter
        values affect the model's metric score. It creates a grid of subplots,
        where each subplot visualizes the relationship between a specific
        hyperparameter and the performance metric, automatically detecting
        whether to use a scatter plot (for continuous) or a box plot
        (for categorical/discrete).

        Args:
            algorithm_name (str): The name of the algorithm to analyze (e.g.,
                'RandomForestClassifier').
            hyperparameters (List[str]): A list of hyperparameter names to plot.
            metric (str, optional): The performance metric for the y-axis.
                Defaults to 'auc'.
            figsize (Optional[Tuple[int, int]], optional): The overall figure
                size. If None, a default is calculated. Defaults to None.
        """
        algo_data = self.clean_data[self.clean_data['algorithm_name'] == algorithm_name].copy()
        
        if algo_data.empty:
            available_algos = self.get_available_algorithms()
            print(f"No data found for algorithm: {algorithm_name}")
            print(f"Available algorithms: {available_algos}")
            return

        # Cap the number of plots to prevent excessively large figures
        MAX_PLOTS = 9
        if len(hyperparameters) > MAX_PLOTS:
            warnings.warn(
                f"Plotting for {len(hyperparameters)} hyperparameters. To avoid an overly large figure, "
                f"only the first {MAX_PLOTS} will be displayed. "
                f"Consider passing a smaller list of hyperparameters if you need to see others.",
                stacklevel=2
            )
            hyperparameters = hyperparameters[:MAX_PLOTS]

        n_params = len(hyperparameters)
        if n_params == 0:
            print(f"No valid hyperparameters were provided to plot for algorithm '{algorithm_name}'.")
            return

        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        fig_size = figsize or (cols * 6, rows * 5)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, param in enumerate(hyperparameters):
            ax = axes[i]
            self._plot_single_performance_vs_hyperparameter(ax, algo_data, param, metric)

        # Hide any unused subplots
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'Performance vs. Hyperparameters for {algorithm_name}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_single_performance_vs_hyperparameter(self, ax: plt.Axes, algo_data: pd.DataFrame, hyperparameter: str, metric: str):
        """Helper to plot performance vs a single hyperparameter on a given axis.

        Args:
            ax (plt.Axes): The matplotlib axis to plot on.
            algo_data (pd.DataFrame): The data for the specific algorithm.
            hyperparameter (str): The name of the hyperparameter to plot.
            metric (str): The name of the performance metric.
        """
        # Extract hyperparameter value for each run
        plot_data = algo_data.copy()
        plot_data[hyperparameter] = plot_data['params_dict'].apply(lambda p: p.get(hyperparameter) if p else None)
        plot_data = plot_data.dropna(subset=[hyperparameter, metric])

        if plot_data.empty:
            ax.text(0.5, 0.5, f"No data for\n'{hyperparameter}'", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(hyperparameter, fontsize=11)
            return

        # Determine if the hyperparameter is numeric or categorical
        param_values = plot_data[hyperparameter]
        is_numeric = pd.api.types.is_numeric_dtype(param_values)
        is_float = pd.api.types.is_float_dtype(param_values)

        # Treat as continuous if it's a float, or an integer with many unique values
        if is_numeric and (is_float or param_values.nunique() > 8):
            # Use log scale for wide ranges
            if param_values.min() > 0 and param_values.max() / param_values.min() > 100:
                ax.set_xscale('log')
                x_label = f'{hyperparameter} (log scale)'
            else:
                x_label = hyperparameter

            # Check if outcome_variable exists for coloring
            scatter_kwargs = {"alpha": 0.6}
            if 'outcome_variable' in plot_data.columns and plot_data['outcome_variable'].nunique() > 1:
                n_outcomes = plot_data['outcome_variable'].nunique()
                if n_outcomes <= 10:
                    scatter_kwargs['hue'] = 'outcome_variable'
                    scatter_kwargs['style'] = 'outcome_variable'
                else:
                    warnings.warn(
                        f"Number of unique outcomes ({n_outcomes}) exceeds the limit of 10 for color encoding. "
                        "Plotting without outcome-based colors.",
                        stacklevel=3
                    )
            
            sns.scatterplot(data=plot_data, x=hyperparameter, y=metric, ax=ax, **scatter_kwargs)
            
            if 'hue' in scatter_kwargs:
                ax.legend(title='Outcome', fontsize='small')
            
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_title(f'{metric.upper()} vs. {hyperparameter}', fontsize=11, fontweight='bold')

        else:  # Categorical or discrete numeric
            if is_numeric:
                # Sort numerically for discrete numeric types
                order = sorted(param_values.unique())
                sns.boxplot(data=plot_data, x=hyperparameter, y=metric, order=order, ax=ax)
            else:
                # Sort alphabetically for categorical types
                param_values_str = param_values.astype(str)
                plot_data[hyperparameter + '_str'] = param_values_str
                order = sorted(param_values_str.unique())
                sns.boxplot(data=plot_data, x=hyperparameter + '_str', y=metric, order=order, ax=ax)
            
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            plt.setp(ax.get_xticklabels(), ha='right')
            ax.set_xlabel(hyperparameter, fontsize=10)
            ax.set_title(f'{metric.upper()} by {hyperparameter}', fontsize=11, fontweight='bold')

        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.grid(True, alpha=0.3)

    def _get_continuous_hyperparameter_correlations(self,
                                                    algorithm_name: str,
                                                    metric: str,
                                                    method: str = 'pearson') -> Optional[pd.DataFrame]:
        """Helper to calculate correlations between continuous hyperparameters and a metric.

        Args:
            algorithm_name (str): The name of the algorithm to analyze.
            metric (str): The performance metric to correlate against.
            method (str, optional): The correlation method ('pearson' or
                'spearman'). Defaults to 'pearson'.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with correlation results, or
            None if no continuous hyperparameters are found.
        """
        algo_data = self.clean_data[self.clean_data['algorithm_name'] == algorithm_name].copy()
        if algo_data.empty:
            return None

        all_params = set().union(*(d.keys() for d in algo_data['params_dict'] if d))
        
        correlations = []
        for param in sorted(list(all_params)):
            param_values = algo_data['params_dict'].apply(lambda p: p.get(param) if p else None).dropna()
            
            is_numeric = pd.api.types.is_numeric_dtype(param_values)
            is_float = pd.api.types.is_float_dtype(param_values)
            
            # Consider a hyperparameter continuous if it's float, or an integer with many unique values
            if not param_values.empty and is_numeric and (is_float or param_values.nunique() > 8):
                # Create a temporary DataFrame for correlation calculation
                temp_df = pd.DataFrame({
                    'param': param_values,
                    'metric': algo_data.loc[param_values.index, metric]
                }).dropna()

                if len(temp_df) < 2:
                    continue

                if method == 'pearson':
                    correlation, p_value = stats.pearsonr(temp_df['param'], temp_df['metric'])
                elif method == 'spearman':
                    correlation, p_value = stats.spearmanr(temp_df['param'], temp_df['metric'])
                else:
                    raise ValueError("Correlation method must be 'pearson' or 'spearman'.")

                if not pd.isna(correlation):
                    correlations.append({
                        'hyperparameter': param,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation),
                        'p_value': p_value,
                        'n_samples': len(temp_df)
                    })
        
        if not correlations:
            return None
            
        return pd.DataFrame(correlations)

    def plot_hyperparameter_importance(self,
                                       algorithm_name: str,
                                       metric: str = 'auc',
                                       top_n_percent: int = 20,
                                       figsize: Optional[Tuple[int, int]] = None):
        """Plots hyperparameter distributions for top models vs. all models.

        This method provides insight into which hyperparameter values are more
        prevalent in high-performing models compared to the overall distribution
        of values explored during the search.

        Args:
            algorithm_name (str): The name of the algorithm to analyze.
            metric (str, optional): The metric used to define "top" models.
                Defaults to 'auc'.
            top_n_percent (int, optional): The percentage of top models to
                compare against. Defaults to 20.
            figsize (Optional[Tuple[int, int]], optional): The figure size for
                the plot. Defaults to None.
        """
        algo_data = self.clean_data[self.clean_data['algorithm_name'] == algorithm_name].copy()
        
        if algo_data.empty:
            available_algos = self.get_available_algorithms()
            print(f"No data found for algorithm: {algorithm_name}")
            print(f"Available algorithms: {available_algos}")
            return

        # Check if metric exists
        if metric not in algo_data.columns:
            print(f"Metric '{metric}' not found. Available metrics: {algo_data.select_dtypes(include=[np.number]).columns.tolist()}")
            return

        # Identify top models
        threshold = algo_data[metric].quantile(1 - (top_n_percent / 100.0))
        top_models = algo_data[algo_data[metric] >= threshold]

        if top_models.empty:
            print(f"No models found in the top {top_n_percent}% for algorithm '{algorithm_name}'.")
            return

        # Get all hyperparameters
        all_params = set()
        for params in algo_data['params_dict']:
            if params:
                all_params.update(params.keys())
        
        hyperparameters = sorted(all_params)
        
        if not hyperparameters:
            print(f"No hyperparameters found for {algorithm_name}.")
            return
        
        n_params = len(hyperparameters)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        fig_size = figsize or (cols * 5, rows * 4)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, param in enumerate(hyperparameters):
            ax = axes[i]
            
            # Extract param values
            all_values = algo_data['params_dict'].apply(lambda p: p.get(param) if p else None).dropna()
            top_values = top_models['params_dict'].apply(lambda p: p.get(param) if p else None).dropna()

            if all_values.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(param, fontsize=11)
                continue

            is_numeric = pd.api.types.is_numeric_dtype(all_values)
            is_float = pd.api.types.is_float_dtype(all_values)

            # Case 1: Continuous numeric data -> Scatter plot of value vs. metric
            # Treat as continuous if it's a float, or an integer with many unique values
            if is_numeric and (is_float or all_values.nunique() > 8):
                # Add the hyperparameter as a column to the dataframe for easy plotting
                plot_df = algo_data.copy()
                plot_df[param] = plot_df['params_dict'].apply(lambda p: p.get(param) if p else None)
                plot_df = plot_df.dropna(subset=[param, metric])

                if plot_df.empty:
                    ax.text(0.5, 0.5, 'No Numeric Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(param, fontsize=11)
                    continue

                # Add a column to distinguish top models
                plot_df['is_top'] = plot_df[metric] >= threshold
                
                sns.scatterplot(data=plot_df, x=param, y=metric, hue='is_top', ax=ax, alpha=0.7, style='is_top', s=50)
                
                ax.set_title(f'{metric.upper()} vs. {param}', fontsize=11, fontweight='bold')
                ax.set_xlabel(param)
                ax.set_ylabel(metric.upper())

                # Use log scale if range is large
                if not plot_df.empty and plot_df[param].min() > 0 and plot_df[param].max() / plot_df[param].min() > 100:
                    ax.set_xscale('log')
                    ax.set_xlabel(f'{param} (log scale)')
                
                # Customize legend
                leg = ax.get_legend()
                if leg:
                    leg.set_title('Performance Tier')
                    for t in leg.get_texts():
                        if t.get_text() == 'False': t.set_text(f'Bottom {100-top_n_percent}%')
                        if t.get_text() == 'True': t.set_text(f'Top {top_n_percent}%')
            # Case 2: Discrete numeric or Categorical data
            else:
                # Sub-case: Discrete numeric data -> Line plot to show trend
                if is_numeric:
                    all_counts = all_values.value_counts(normalize=True)
                    top_counts = top_values.value_counts(normalize=True)
                    df_plot = pd.concat([all_counts.rename('All'), top_counts.rename('Top')],
                                      axis=1).fillna(0).sort_index()
                    
                    df_plot.plot(kind='line', marker='o', ax=ax, linestyle='-')
                    ax.set_title(f'{param}', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Proportion')
                    ax.set_xlabel(param)
                    # Ensure all discrete ticks are shown and formatted nicely
                    ax.set_xticks(df_plot.index) # Ensure all discrete ticks are shown
                    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:g}')) # Use general format for ticks
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') # Correctly set rotation and alignment
                # Sub-case: Categorical data -> Bar plot
                else:
                    all_counts = all_values.value_counts(normalize=True)
                    top_counts = top_values.value_counts(normalize=True)
                    df_plot = pd.concat([all_counts.rename('All'), top_counts.rename('Top')],
                                      axis=1).fillna(0).sort_index()
                    df_plot.plot(kind='bar', ax=ax, width=0.8)
                    ax.set_title(f'{param}', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Proportion')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                ax.legend()

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'Hyperparameter Analysis: {algorithm_name}\n(All vs. Top {top_n_percent}% by {metric.upper()})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_hyperparameter_correlations(self,
                                        algorithm_name: str,
                                        metric: str = 'auc',
                                        method: str = 'pearson',
                                        figsize: Optional[Tuple[int, int]] = None,
                                        show_correlation_stats: bool = True):
        """Plots correlation between continuous hyperparameters and a performance metric.

        This method creates scatter plots to visualize the relationship between
        each continuous hyperparameter and the target metric, including a
        regression line and correlation statistics.

        Args:
            algorithm_name (str): The name of the algorithm to analyze.
            metric (str, optional): The performance metric. Defaults to 'auc'.
            method (str, optional): The correlation method ('pearson' or 'spearman').
                Defaults to 'pearson'.
            figsize (Optional[Tuple[int, int]], optional): The figure size.
                Defaults to None.
            show_correlation_stats (bool, optional): Whether to print a summary
                table of correlations. Defaults to True.
        """
        algo_data = self.clean_data[self.clean_data['algorithm_name'] == algorithm_name].copy()
        
        if algo_data.empty:
            available_algos = self.get_available_algorithms()
            print(f"No data found for algorithm: {algorithm_name}")
            print(f"Available algorithms: {available_algos}")
            return

        # Check if metric exists
        if metric not in algo_data.columns:
            print(f"Metric '{metric}' not found. Available metrics: {algo_data.select_dtypes(include=[np.number]).columns.tolist()}")
            return

        if method not in ['pearson', 'spearman']:
            raise ValueError("Method must be 'pearson' or 'spearman'")

        # Get correlations
        correlation_results_df = self._get_continuous_hyperparameter_correlations(algorithm_name, metric, method)

        if correlation_results_df is None or correlation_results_df.empty:
            print(f"No continuous hyperparameters found for {algorithm_name}.")
            return
        
        n_params = len(correlation_results_df)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        fig_size = figsize or (cols * 6, rows * 5)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, row in correlation_results_df.iterrows():
            param = row['hyperparameter']
            ax = axes[i]
            
            # Extract param values and create plotting dataframe
            plot_df = algo_data.copy()
            plot_df[param] = plot_df['params_dict'].apply(lambda p: p.get(param) if p else None)
            plot_df = plot_df.dropna(subset=[param, metric])

            if plot_df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(param, fontsize=12)
                continue

            x_values = plot_df[param]
            y_values = plot_df[metric]

            # Calculate correlation statistics
            correlation = row['correlation']
            p_value = row['p_value']
            
            # Create scatter plot
            sns.scatterplot(data=plot_df, x=param, y=metric, ax=ax, alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_values.min(), x_values.max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Use log scale if range is large
            if x_values.min() > 0 and x_values.max() / x_values.min() > 100:
                ax.set_xscale('log')
                param_label = f'{param} (log scale)'
            else:
                param_label = param
            
            if method == 'pearson':
                corr_label = 'r'
            else:
                corr_label = 'ρ'

            # Create title with correlation info
            if show_correlation_stats:
                title = f'{param}\n{corr_label} = {correlation:.3f}'
                if p_value < 0.001:
                    title += ' (p < 0.001)'
                elif p_value < 0.01:
                    title += f' (p < 0.01)'
                elif p_value < 0.05:
                    title += f' (p = {p_value:.3f})'
                else:
                    title += f' (p = {p_value:.3f})'
            else:
                title = param
                
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel(param_label, fontsize=10)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'Hyperparameter Correlations with {metric.upper()}: {algorithm_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # Print correlation summary
        if show_correlation_stats:
            print(f"\nCorrelation Summary for {algorithm_name}:")
            print("-" * 60)
            corr_df = correlation_results_df.sort_values('abs_correlation', ascending=False)
            if method == 'pearson':
                corr_label = 'r'
            else:
                corr_label = 'ρ'
            
            for _, row in corr_df.iterrows():
                significance = ""
                if row['p_value'] < 0.001:
                    significance = "***"
                elif row['p_value'] < 0.01:
                    significance = "**"
                elif row['p_value'] < 0.05:
                    significance = "*"
                
                print(f"{row['hyperparameter']:20s}: {corr_label} = {row['correlation']:6.3f}{significance:3s} "
                      f"(p = {row['p_value']:.3f}, n = {row['n_samples']})")
            
            print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

    def plot_top_correlations(self,
                             algorithm_name: str,
                             metric: str = 'auc',
                             method: str = 'pearson',
                             top_n: int = 5,
                             figsize: Tuple[int, int] = (15, 10)):
        """Plots only the top N most correlated hyperparameters with the metric.

        Args:
            algorithm_name (str): The name of the algorithm to analyze.
            metric (str, optional): The performance metric. Defaults to 'auc'.
            method (str, optional): The correlation method ('pearson' or 'spearman').
                Defaults to 'pearson'.
            top_n (int, optional): The number of top correlated hyperparameters
                to plot. Defaults to 5.
            figsize (Tuple[int, int], optional): The figure size.
                Defaults to (15, 10).
        """
        algo_data = self.clean_data[self.clean_data['algorithm_name'] == algorithm_name].copy()
        
        if algo_data.empty:
            available_algos = self.get_available_algorithms()
            print(f"No data found for algorithm: {algorithm_name}")
            print(f"Available algorithms: {available_algos}")
            return

        if method not in ['pearson', 'spearman']:
            raise ValueError("Method must be 'pearson' or 'spearman'")

        # Get correlations and take top N
        correlations_df = self._get_continuous_hyperparameter_correlations(algorithm_name, metric, method)
        
        if correlations_df is None or correlations_df.empty:
            print(f"No continuous hyperparameters found for {algorithm_name} to plot correlations.")
            return
        top_correlations = correlations_df.sort_values('abs_correlation', ascending=False).head(top_n)
        
        # Plot top correlations
        n_plots = len(top_correlations)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, (_, row) in enumerate(top_correlations.iterrows()):
            ax = axes[i]
            param = row['hyperparameter']
            
            # Create plotting dataframe
            plot_df = algo_data.copy()
            plot_df[param] = plot_df['params_dict'].apply(lambda p: p.get(param) if p else None)
            plot_df = plot_df.dropna(subset=[param, metric])
            
            x_values = plot_df[param]
            y_values = plot_df[metric]
            
            # Create scatter plot with color coding by performance
            scatter = sns.scatterplot(data=plot_df, x=param, y=metric, ax=ax, 
                                     c=plot_df[metric], cmap='viridis', alpha=0.7, s=60)
            
            # Add trend line
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_values.min(), x_values.max(), 100)
            ax.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=3)
            
            # Use log scale if needed
            if x_values.min() > 0 and x_values.max() / x_values.min() > 100:
                ax.set_xscale('log')
                param_label = f'{param} (log scale)'
            else:
                param_label = param
            
            # Title with ranking and correlation
            significance = ""
            if row['p_value'] < 0.001:
                significance = "***"
            elif row['p_value'] < 0.01:
                significance = "**"
            elif row['p_value'] < 0.05:
                significance = "*"
            
            if method == 'pearson':
                corr_label = 'r'
            else:
                corr_label = 'ρ'
            
            title = f'#{i+1}: {param}\n{corr_label} = {row["correlation"]:.3f}{significance}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(param_label, fontsize=11)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for the first plot
            if i == 0:
                cbar = plt.colorbar(scatter.collections[0], ax=ax)
                cbar.set_label(metric.upper(), fontsize=10)

        # Hide unused subplots
        for j in range(len(top_correlations), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'Top {min(top_n, len(top_correlations))} Hyperparameter Correlations: {algorithm_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        return top_correlations