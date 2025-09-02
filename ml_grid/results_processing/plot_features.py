# plot_features.py
"""
Feature analysis and importance plotting module for ML results analysis.
Focuses on feature usage and impact on performance, with outcome stratification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import warnings

# Maximum number of outcomes to display in stratified plots to avoid clutter.
MAX_OUTCOMES_FOR_STRATIFIED_PLOT = 10
# Maximum number of features to analyze in detail to avoid performance issues.
MAX_FEATURES_FOR_ANALYSIS = 500
# Maximum number of features to show in an UpSet plot matrix for readability.
MAX_FEATURES_FOR_UPSET = 40

try:
    from upsetplot import from_memberships, UpSet
except ImportError:
    UpSet = None
    warnings.warn("`upsetplot` library not found. `plot_feature_set_intersections` will be unavailable. "
                  "Install with: pip install upsetplot", stacklevel=2)

from ml_grid.results_processing.core import get_clean_data


class FeatureAnalysisPlotter:
    """
    Class for creating feature analysis and importance visualizations.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the feature analysis plotter.

        Args:
            data: Results DataFrame, must contain 'decoded_features' column.
        """
        if 'decoded_features' not in data.columns:
            raise ValueError("Data must contain a 'decoded_features' column. Ensure ResultsAggregator was run with a feature names CSV.")
        
        self.data = data
        self.clean_data = get_clean_data(data)
        
        # Explode the features for easier analysis. Drop rows where feature is NaN (from empty lists)
        self.feature_df = self.clean_data.explode('decoded_features').rename(columns={'decoded_features': 'feature'}).dropna(subset=['feature'])
        
        plt.style.use('default')
        sns.set_palette("viridis")

    def plot_feature_usage_frequency(self, top_n: int = 20,
                                     stratify_by_outcome: bool = False,
                                     outcomes_to_plot: Optional[List[str]] = None,
                                     figsize: Optional[Tuple[int, int]] = None):
        """
        Plots the frequency of each feature's usage in successful runs.

        Args:
            top_n: Number of most frequent features to plot.
            stratify_by_outcome: If True, create separate plots for each outcome.
            outcomes_to_plot: Specific outcomes to plot (if stratified).
            figsize: Figure size for the plot. If None, a default is calculated.
        """
        if not stratify_by_outcome:
            fig_size = figsize or (10, 8)
            self._plot_single_feature_frequency(top_n, fig_size)
        else:
            if 'outcome_variable' not in self.clean_data.columns:
                raise ValueError("outcome_variable column not found for stratification.")
            self._plot_stratified_feature_frequency(top_n, outcomes_to_plot, figsize)

    def _plot_single_feature_frequency(self, top_n: int, figsize: Tuple[int, int]):
        """Helper for plotting overall feature frequency."""
        plt.figure(figsize=figsize)
        
        feature_counts = self.feature_df['feature'].value_counts().nlargest(top_n)
        
        sns.barplot(x=feature_counts.values, y=feature_counts.index, hue=feature_counts.index, orient='h', palette='viridis', legend=False)
        
        plt.title(f'Top {top_n} Most Frequently Used Features (All Outcomes)', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Times Used in Successful Runs', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

    def _plot_stratified_feature_frequency(self, top_n: int, outcomes_to_plot: Optional[List[str]], figsize: Optional[Tuple[int, int]]):
        """Helper for plotting stratified feature frequency."""
        outcomes = outcomes_to_plot or sorted(self.clean_data['outcome_variable'].unique())
        if len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT and outcomes_to_plot is None:
            warnings.warn(
                f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Displaying the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                "Use the 'outcomes_to_plot' parameter to select specific outcomes.",
                stacklevel=2
            )
            outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        n_outcomes = len(outcomes)
        
        cols = min(2, n_outcomes)
        rows = (n_outcomes + cols - 1) // cols
        
        fig_size = figsize or (cols * 7, rows * 6)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        axes = axes.flatten()

        for i, outcome in enumerate(outcomes):
            ax = axes[i]
            outcome_feature_df = self.feature_df[self.feature_df['outcome_variable'] == outcome]
            
            if not outcome_feature_df.empty:
                feature_counts = outcome_feature_df['feature'].value_counts().nlargest(top_n)
                
                if not feature_counts.empty:
                    sns.barplot(x=feature_counts.values, y=feature_counts.index, hue=feature_counts.index, orient='h', ax=ax, palette='plasma', legend=False)
                    ax.set_title(f'{outcome} - Top {min(top_n, len(feature_counts))} Features', fontsize=11, fontweight='bold')
                    ax.set_xlabel('Usage Count')
                    ax.set_ylabel('')
                else:
                    ax.text(0.5, 0.5, 'No Feature Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{outcome}', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{outcome}', fontsize=11)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.suptitle(f'Top {top_n} Most Frequently Used Features per Outcome', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_feature_performance_impact(self, metric: str = 'auc',
                                        outcomes: Optional[List[str]] = None,
                                        top_n: int = 20,
                                        min_usage: int = 5,
                                        top_n_features_to_consider: int = MAX_FEATURES_FOR_ANALYSIS,
                                        figsize_per_outcome: Tuple[int, int] = (10, 8)):
        """
        Plots the impact of features on a given performance metric for each outcome.

        Impact is calculated as:
        (Mean metric of runs WITH the feature) - (Mean metric of runs WITHOUT the feature)

        Args:
            metric: The performance metric to evaluate (e.g., 'auc', 'f1').
            outcomes: List of outcome variables to plot. If None, all are plotted.
            top_n: Number of top positive and negative impacting features to show.
            min_usage: Minimum number of times a feature must be used to be included.
            top_n_features_to_consider: Max number of most frequent features to analyze for impact.
            figsize_per_outcome: The figure size for each individual outcome plot.
        """
        if 'outcome_variable' not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found for this analysis.")
        
        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")
            
        outcomes_to_plot = outcomes or sorted(self.clean_data['outcome_variable'].unique())

        if len(outcomes_to_plot) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT and outcomes is None:
            warnings.warn(
                f"Found {len(outcomes_to_plot)} outcomes, which is more than the plot limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Generating plots for the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT} outcomes. "
                "Use the 'outcomes' parameter to select specific outcomes.",
                stacklevel=2
            )
            outcomes_to_plot = outcomes_to_plot[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        for outcome in outcomes_to_plot:
            outcome_data = self.clean_data[self.clean_data['outcome_variable'] == outcome].copy()
            
            if outcome_data.empty:
                warnings.warn(f"Skipping plot for outcome '{outcome}': No successful data available.", stacklevel=2)
                continue

            # Check if 'decoded_features' column is populated for this outcome's data
            # Count rows where 'decoded_features' is a non-empty list
            non_empty_decoded_features_count = outcome_data['decoded_features'].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            ).sum()

            if non_empty_decoded_features_count == 0:
                warnings.warn(
                    f"Skipping plot for outcome '{outcome}': No valid feature lists found. "
                    f"This may be due to a feature name mismatch during data loading or because "
                    f"all feature lists for this outcome are all-zeros.",
                    stacklevel=2
                )
                continue

            # Get all unique features used for this specific outcome
            outcome_feature_df = self.feature_df[self.feature_df['outcome_variable'] == outcome]
            all_features_in_outcome = outcome_feature_df['feature'].unique()

            # Pre-filter features for performance if there are too many
            features_to_analyze = all_features_in_outcome
            if len(all_features_in_outcome) > top_n_features_to_consider:
                warnings.warn(
                    f"Outcome '{outcome}' has {len(all_features_in_outcome)} unique features. "
                    f"To improve performance, analysis is limited to the top {top_n_features_to_consider} most frequently used features. "
                    "You can change this limit with the 'top_n_features_to_consider' parameter.",
                    stacklevel=2
                )
                # Get the most frequent features for this outcome
                feature_counts = outcome_feature_df['feature'].value_counts()
                features_to_analyze = feature_counts.nlargest(top_n_features_to_consider).index.tolist()

            impact_data = []
            for feature in features_to_analyze:
                # Create a boolean mask for runs containing the feature
                has_feature_mask = outcome_data['decoded_features'].apply(lambda x: feature in x if isinstance(x, list) else False)
                
                usage_count = has_feature_mask.sum()
                if usage_count < min_usage or usage_count == len(outcome_data):
                    # Skip if feature is used too little or in all runs (no 'without' group)
                    continue
                
                mean_with_feature = outcome_data.loc[has_feature_mask, metric].mean()
                mean_without_feature = outcome_data.loc[~has_feature_mask, metric].mean()
                
                impact = mean_with_feature - mean_without_feature
                
                if not pd.isna(impact):
                    impact_data.append({
                        'feature': feature,
                        'impact': impact,
                    })
            
            if not impact_data: # Final check after iterating through all features
                warnings.warn(f"Skipping plot for outcome '{outcome}': No features met the analysis criteria "
                              f"after filtering (min_usage={min_usage}, always present/absent).", stacklevel=2)
                continue
                
            impact_df = pd.DataFrame(impact_data)
            
            # Get top and bottom N features by impact
            top_impact = impact_df.nlargest(top_n, 'impact')
            bottom_impact = impact_df.nsmallest(top_n, 'impact')
            plot_df = pd.concat([top_impact, bottom_impact]).drop_duplicates().sort_values('impact', ascending=False)

            plt.figure(figsize=figsize_per_outcome)
            
            colors = ['#3a923a' if x > 0 else '#c14242' for x in plot_df['impact']]
            ax = sns.barplot(x='impact', y='feature', data=plot_df, orient='h', palette=colors, hue='feature', legend=False)
            
            ax.set_title(f'Feature Impact on {metric.upper()} for Outcome: {outcome}', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'Change in Mean {metric.upper()} (With vs. Without Feature)', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            
            plt.tight_layout()
            plt.show()

    def plot_feature_metric_correlation(self,
                                        metric: str = 'auc',
                                        outcomes: Optional[List[str]] = None, 
                                        top_n: int = 20,
                                        min_usage: int = 5,
                                        top_n_features_to_consider: int = MAX_FEATURES_FOR_ANALYSIS,
                                        figsize_per_outcome: Tuple[int, int] = (10, 8)):
        """
        Plots the point-biserial correlation between feature presence and a performance metric.

        This shows which features, when present, are most correlated with higher or lower
        performance for a given outcome.

        Args:
            metric: The performance metric to correlate with (e.g., 'auc', 'f1').
            outcomes: List of outcome variables to plot. If None, all are plotted.
            top_n: Number of top positive and negative correlated features to show.
            min_usage: Minimum number of times a feature must be used to be included.
            top_n_features_to_consider: Max number of most frequent features to analyze for correlation.
            figsize_per_outcome: The figure size for each individual outcome plot.
        """
        if 'outcome_variable' not in self.clean_data.columns:
            raise ValueError("outcome_variable column not found for this analysis.")

        if metric not in self.clean_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data.")

        outcomes_to_plot = outcomes or sorted(self.clean_data['outcome_variable'].unique())

        if len(outcomes_to_plot) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT and outcomes is None:
            warnings.warn(
                f"Found {len(outcomes_to_plot)} outcomes, which is more than the plot limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                f"Generating plots for the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT} outcomes. "
                "Use the 'outcomes' parameter to select specific outcomes.",
                stacklevel=2
            )
            outcomes_to_plot = outcomes_to_plot[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

        for outcome in outcomes_to_plot:
            outcome_data = self.clean_data[self.clean_data['outcome_variable'] == outcome].copy()

            if outcome_data.empty:
                warnings.warn(f"Skipping plot for outcome '{outcome}': No successful data available.", stacklevel=2)
                continue

            # Check if 'decoded_features' column is populated
            non_empty_decoded_features_count = outcome_data['decoded_features'].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            ).sum()

            if non_empty_decoded_features_count == 0:
                warnings.warn(
                    f"Skipping plot for outcome '{outcome}': No valid feature lists found.",
                    stacklevel=2
                )
                continue

            # Get all unique features used for this specific outcome
            outcome_feature_df = self.feature_df[self.feature_df['outcome_variable'] == outcome]
            all_features_in_outcome = outcome_feature_df['feature'].unique()
            
            # Pre-filter features for performance if there are too many
            features_to_analyze = all_features_in_outcome
            if len(all_features_in_outcome) > top_n_features_to_consider:
                warnings.warn(
                    f"Outcome '{outcome}' has {len(all_features_in_outcome)} unique features. "
                    f"To improve performance, correlation analysis is limited to the top {top_n_features_to_consider} most frequently used features. "
                    "You can change this limit with the 'top_n_features_to_consider' parameter.",
                    stacklevel=2
                )
                # Get the most frequent features for this outcome
                feature_counts = outcome_feature_df['feature'].value_counts()
                features_to_analyze = feature_counts.nlargest(top_n_features_to_consider).index.tolist()
            
            correlation_data = []
            for feature in features_to_analyze:
                # Create a boolean mask for runs containing the feature
                has_feature_mask = outcome_data['decoded_features'].apply(lambda x: feature in x if isinstance(x, list) else False)

                usage_count = has_feature_mask.sum()
                if usage_count < min_usage or usage_count == len(outcome_data):
                    # Skip if feature is used too little or in all runs (no variance)
                    continue

                # Calculate point-biserial correlation
                correlation = outcome_data[metric].corr(has_feature_mask)

                if not pd.isna(correlation):
                    correlation_data.append({'feature': feature, 'correlation': correlation})

            if not correlation_data:
                warnings.warn(f"Skipping plot for outcome '{outcome}': No features met the analysis criteria (min_usage={min_usage}).", stacklevel=2)
                continue

            correlation_df = pd.DataFrame(correlation_data)

            # Get top and bottom N features by correlation
            top_corr = correlation_df.nlargest(top_n, 'correlation')
            bottom_corr = correlation_df.nsmallest(top_n, 'correlation')
            plot_df = pd.concat([top_corr, bottom_corr]).drop_duplicates().sort_values('correlation', ascending=False)

            plt.figure(figsize=figsize_per_outcome)

            colors = ['#3a923a' if x > 0 else '#c14242' for x in plot_df['correlation']]
            ax = sns.barplot(x='correlation', y='feature', data=plot_df, orient='h', palette=colors, hue='feature', legend=False)

            ax.set_title(f'Feature Correlation with {metric.upper()} for Outcome: {outcome}', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'Point-Biserial Correlation with {metric.upper()}', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlim([-1, 1])  # Correlation is between -1 and 1

            plt.tight_layout()
            plt.show()

    def plot_feature_set_intersections(self,
                                       top_n_sets: int = 10,
                                       min_subset_size: int = 5,
                                       stratify_by_outcome: bool = False,
                                       max_features_for_upset: int = MAX_FEATURES_FOR_UPSET,
                                       figsize: Tuple[int, int] = (12, 7)):
        """
        Plots the intersections of feature sets used in successful models using an UpSet plot.

        This helps visualize which combinations of features are most frequently used together.

        Args:
            top_n_sets: The number of most frequent feature set intersections to plot.
            min_subset_size: The minimum number of models a feature set must appear in to be plotted.
            stratify_by_outcome: If True, create a separate plot for each outcome variable.
            max_features_for_upset: Max number of most frequent features to include in the UpSet plot matrix.
            figsize: The figure size for the plot.
        """
        if UpSet is None:
            warnings.warn("Cannot generate UpSet plot because `upsetplot` is not installed. "
                          "Install with: pip install upsetplot", stacklevel=2)
            return

        if not stratify_by_outcome:
            self._plot_single_upset(self.clean_data, 'All Outcomes', top_n_sets, min_subset_size, max_features_for_upset, figsize)
        else:
            if 'outcome_variable' not in self.clean_data.columns:
                raise ValueError("outcome_variable column not found for stratification.")
            
            outcomes = sorted(self.clean_data['outcome_variable'].unique())
            if len(outcomes) > MAX_OUTCOMES_FOR_STRATIFIED_PLOT:
                warnings.warn(
                    f"Found {len(outcomes)} outcomes, which is more than the display limit of {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                    f"Displaying plots for the first {MAX_OUTCOMES_FOR_STRATIFIED_PLOT}. "
                    "To plot for specific outcomes, filter the input DataFrame before creating the plotter.",
                    stacklevel=2
                )
                outcomes = outcomes[:MAX_OUTCOMES_FOR_STRATIFIED_PLOT]

            for outcome in outcomes:
                outcome_data = self.clean_data[self.clean_data['outcome_variable'] == outcome]
                if not outcome_data.empty:
                    self._plot_single_upset(outcome_data, outcome, top_n_sets, min_subset_size, max_features_for_upset, figsize)

    def _plot_single_upset(self, data: pd.DataFrame, title: str, top_n_sets: int, min_subset_size: int, max_features_for_upset: int, figsize: Tuple[int, int]):
        """Helper to generate a single UpSet plot for given data."""
        # Filter out rows with empty or invalid feature lists
        feature_sets = data['decoded_features'].dropna().apply(lambda x: x if isinstance(x, list) and x else None).dropna()

        if feature_sets.empty:
            warnings.warn(f"No valid feature sets to plot for '{title}'. Skipping.", stacklevel=3)
            return

        # Limit features in the UpSet plot for readability
        # Explode to get all feature occurrences and count them
        all_features_flat = [feature for feature_list in feature_sets for feature in feature_list]
        if not all_features_flat:
             warnings.warn(f"No features found in feature sets for '{title}'. Skipping.", stacklevel=3)
             return
        
        feature_counts = pd.Series(all_features_flat).value_counts()
        
        # If there are more features than the limit, filter the sets
        if len(feature_counts) > max_features_for_upset:
            warnings.warn(
                f"Found {len(feature_counts)} unique features for '{title}'. To keep the UpSet plot readable, "
                f"displaying intersections for the {max_features_for_upset} most frequent features only. "
                f"You can change this with the 'max_features_for_upset' parameter.",
                stacklevel=3
            )
            top_features = set(feature_counts.nlargest(max_features_for_upset).index)
            # Filter each list in the series to only contain top features
            feature_sets = feature_sets.apply(lambda x: [f for f in x if f in top_features])
            # Drop any sets that became empty after filtering
            feature_sets = feature_sets[feature_sets.str.len() > 0]

        if feature_sets.empty:
            warnings.warn(f"No feature sets remaining for '{title}' after filtering for top features. Skipping.", stacklevel=3)
            return

        # Convert the Series of lists to a plain list of lists before passing to from_memberships.
        # This is more robust than passing the Series object directly.
        upset_source_data = from_memberships(feature_sets.tolist())

        if upset_source_data.empty:
            warnings.warn(f"Could not generate upset data from feature sets for '{title}'. Skipping.", stacklevel=3)
            return

        max_intersection_size = upset_source_data.max() if not upset_source_data.empty else 0

        # Apply filtering and sorting before plotting for cleaner logic.
        upset_plot_data = upset_source_data[upset_source_data >= min_subset_size]
        upset_plot_data = upset_plot_data.sort_values(ascending=False).head(top_n_sets)

        if upset_plot_data.empty:
            warnings.warn(
                f"No feature set intersections for '{title}' met the plotting criteria "
                f"(min_subset_size={min_subset_size}). The largest intersection found was of size "
                f"{int(max_intersection_size)}. Consider lowering `min_subset_size` in the function call. "
                f"Skipping plot for '{title}'.",
                stacklevel=3
            )
            return

        fig = plt.figure(figsize=figsize)
        # Data is now pre-filtered and pre-sorted.
        upset = UpSet(upset_plot_data,
                      show_counts=True,
                      # Respect the pre-sorted data order.
                      sort_by=None)
        upset.plot(fig=fig)
        plt.suptitle(f'Feature Set Intersections - {title}', fontsize=16, fontweight='bold')
        plt.show()