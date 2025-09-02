# plot_master.py
"""
Master plotting module that provides a single entry point to generate a
comprehensive set of visualizations for ML results analysis.
"""

import pandas as pd
from typing import List, Optional

# Import all the individual plotter classes
from .plot_algorithms import AlgorithmComparisonPlotter
from .plot_distributions import DistributionPlotter
from .plot_features import FeatureAnalysisPlotter
from .plot_timeline import TimelineAnalysisPlotter
from .plot_hyperparameters import HyperparameterAnalysisPlotter
from .plot_feature_categories import FeatureCategoryPlotter
from .plot_pipeline_parameters import PipelineParameterPlotter
from .plot_global_importance import GlobalImportancePlotter
from .plot_interactions import InteractionPlotter


class MasterPlotter:
    """
    A facade class that orchestrates various specialized plotters to generate a
    full suite of analysis plots from ML results data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the MasterPlotter with the aggregated results data.

        Args:
            data: A pandas DataFrame containing the aggregated ML experiment results.
                  Must be a non-empty DataFrame.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")

        self.data = data

        # Instantiate all the specialized plotters
        self.algo_plotter = AlgorithmComparisonPlotter(self.data)
        self.dist_plotter = DistributionPlotter(self.data)
        self.timeline_plotter = TimelineAnalysisPlotter(self.data)
        self.interaction_plotter = InteractionPlotter(self.data)

        # Global importance plotter
        try:
            self.global_importance_plotter = GlobalImportancePlotter(self.data)
        except ValueError as e:
            self.global_importance_plotter = None
            print(f"Warning: Could not initialize GlobalImportancePlotter. Reason: {e}")

        # Pipeline parameter plotter
        try:
            self.pipeline_plotter = PipelineParameterPlotter(self.data)
        except ValueError as e:
            self.pipeline_plotter = None
            print(f"Warning: Could not initialize PipelineParameterPlotter. Reason: {e}")

        # Feature category plotter
        try:
            self.feature_cat_plotter = FeatureCategoryPlotter(self.data)
        except ValueError as e:
            self.feature_cat_plotter = None
            print(f"Warning: Could not initialize FeatureCategoryPlotter. Reason: {e}")

        # Feature plotter requires 'decoded_features' column
        if 'decoded_features' in self.data.columns:
            self.feature_plotter = FeatureAnalysisPlotter(self.data)
        else:
            self.feature_plotter = None
            print("Warning: 'decoded_features' column not found. Feature-related plots will be skipped.")

        # Hyperparameter plotter requires 'algorithm_implementation' column
        if 'algorithm_implementation' in self.data.columns:
            self.hyperparam_plotter = HyperparameterAnalysisPlotter(self.data)
        else:
            self.hyperparam_plotter = None
            print("Warning: 'algorithm_implementation' column not found. Hyperparameter-related plots will be skipped.")

    def plot_all(self,
                 metric: str = 'auc_m',
                 stratify_by_outcome: bool = True,
                 top_n_features: int = 20,
                 top_n_algorithms: int = 10):
        """
        Generates a comprehensive set of standard plots from all available plotters.

        This method calls the main plotting functions from each specialized plotter
        to provide a full overview of the results, including algorithm comparisons,
        metric distributions, timeline trends, and feature importance.

        Args:
            metric: The primary performance metric to use for plotting (e.g., 'auc', 'f1').
            stratify_by_outcome: If True, creates plots stratified by the 'outcome_variable' column.
            top_n_features: The number of top features to show in feature-related plots.
            top_n_algorithms: The number of top algorithms to show in ranking plots.
        """
        print(f"--- Generating All Plots (Metric: {metric.upper()}, Stratified: {stratify_by_outcome}) ---")

        print("\n>>> 1. Generating Algorithm Comparison Plots...")
        try:
            self.algo_plotter.plot_algorithm_boxplots(metric=metric, stratify_by_outcome=stratify_by_outcome)
            self.algo_plotter.plot_algorithm_performance_heatmap(metric=metric, aggregation='mean')
            self.algo_plotter.plot_algorithm_ranking(metric=metric, stratify_by_outcome=stratify_by_outcome, top_n=top_n_algorithms)
            self.algo_plotter.plot_algorithm_stability(metric=metric, top_n=top_n_algorithms)
            self.algo_plotter.plot_performance_tradeoff(metric_y=metric, metric_x='run_time', top_n_algos=top_n_algorithms)
            self.algo_plotter.plot_pareto_front(metric_y=metric, metric_x='run_time')
            self.algo_plotter.plot_statistical_significance_heatmap(metric=metric)
        except Exception as e:
            print(f"Warning: Could not generate algorithm plots. Reason: {e}")

        print("\n>>> 2. Generating Distribution Plots...")
        try:
            self.dist_plotter.plot_metric_distributions(metrics=[metric, 'f1', 'mcc'], stratify_by_outcome=stratify_by_outcome)
            self.dist_plotter.plot_comparative_distributions(metric=metric, plot_type='violin')
        except Exception as e:
            print(f"Warning: Could not generate distribution plots. Reason: {e}")

        print("\n>>> 3. Generating Timeline Plots...")
        try:
            self.timeline_plotter.plot_performance_timeline(metric=metric, stratify_by_outcome=stratify_by_outcome)
            self.timeline_plotter.plot_improvement_trends(metric=metric, stratify_by_outcome=stratify_by_outcome)
            self.timeline_plotter.plot_computational_cost_timeline(stratify_by_outcome=stratify_by_outcome)
        except Exception as e:
            print(f"Warning: Could not generate timeline plots. Reason: {e}")

        if self.feature_plotter:
            print("\n>>> 4. Generating Feature Analysis Plots...")
            try:
                self.feature_plotter.plot_feature_usage_frequency(top_n=top_n_features, stratify_by_outcome=stratify_by_outcome)
                self.feature_plotter.plot_feature_performance_impact(metric=metric, top_n=top_n_features // 2)
                self.feature_plotter.plot_feature_metric_correlation(metric=metric, top_n=top_n_features // 2)
                self.feature_plotter.plot_feature_set_intersections(top_n_sets=15, stratify_by_outcome=stratify_by_outcome)
            except Exception as e:
                print(f"Warning: Could not generate feature plots. Reason: {e}")

        if self.hyperparam_plotter:
            print("\n>>> 5. Generating Hyperparameter Analysis Plots...")
            try:
                # Get algorithms that have parsable hyperparameters
                available_algos = self.hyperparam_plotter.get_available_algorithms()
                if not available_algos:
                    print("Info: No algorithms with parsable hyperparameters found. Skipping hyperparameter plots.")
                else:
                    # Find the top overall algorithm *that has parsable hyperparameters*
                    plottable_data = self.data[self.data['method_name'].isin(available_algos)]

                    if plottable_data.empty:
                        print("Info: No data found for algorithms with parsable hyperparameters. Skipping.")
                    else:
                        top_algo = plottable_data.groupby('method_name')[metric].mean().idxmax()
                        print(f"Analyzing hyperparameters for top algorithm: {top_algo}")

                        self.hyperparam_plotter.plot_hyperparameter_importance(algorithm_name=top_algo, metric=metric)

                        # For the second plot, find a hyperparameter of the top algorithm to analyze
                        # This uses the parsed data within the hyperparam_plotter instance
                        algo_data = self.hyperparam_plotter.clean_data[self.hyperparam_plotter.clean_data['algorithm_name'] == top_algo]
                        
                        # Get all hyperparameters for this algorithm from the first valid param dict
                        params_list = [p for p in algo_data['params_dict'] if p]
                        if params_list:
                            # Find all hyperparameters that have more than one unique value to make interesting plots
                            hyperparameters_to_plot = []
                            all_keys = set().union(*(d.keys() for d in params_list))
                            for param in sorted(list(all_keys)):
                                if algo_data['params_dict'].apply(lambda p: p.get(param) if p else None).nunique() > 1:
                                    hyperparameters_to_plot.append(param)
                            
                            if hyperparameters_to_plot:
                                 self.hyperparam_plotter.plot_performance_by_hyperparameter(
                                     algorithm_name=top_algo, 
                                     hyperparameters=hyperparameters_to_plot, 
                                     metric=metric
                                 )
                            else:
                                print(f"Info: No hyperparameters with multiple values found for {top_algo} to generate performance plot.")
            except Exception as e:
                print(f"Warning: Could not generate hyperparameter plots. Reason: {e}")

        if self.feature_cat_plotter:
            print("\n>>> 6. Generating Feature Category Analysis Plots...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                category_metric = 'auc' if 'auc' in self.data.columns else metric
                print(f"Analyzing feature category impact on metric: {category_metric.upper()}")
                self.feature_cat_plotter.plot_category_performance_boxplots(metric=category_metric)
                self.feature_cat_plotter.plot_category_impact_on_metric(metric=category_metric)
            except Exception as e:
                print(f"Warning: Could not generate feature category plots. Reason: {e}")

        if self.pipeline_plotter:
            print("\n>>> 7. Generating Pipeline Parameter Analysis Plots...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                pipeline_metric = 'auc' if 'auc' in self.data.columns else metric
                print(f"Analyzing pipeline parameter impact on metric: {pipeline_metric.upper()}")
                self.pipeline_plotter.plot_categorical_parameters(metric=pipeline_metric)
                self.pipeline_plotter.plot_continuous_parameters(metric=pipeline_metric)
            except Exception as e:
                print(f"Warning: Could not generate pipeline parameter plots. Reason: {e}")

        if self.global_importance_plotter:
            print("\n>>> 8. Generating Global Importance Analysis Plot...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                global_metric = 'auc' if 'auc' in self.data.columns else metric
                self.global_importance_plotter.plot_global_importance(metric=global_metric, top_n=40)
            except Exception as e:
                print(f"Warning: Could not generate global importance plot. Reason: {e}")

        if self.interaction_plotter:
            print("\n>>> 9. Generating Interaction Analysis Plots...")
            try:
                # Example interaction plot
                self.interaction_plotter.plot_categorical_interaction(param1='resample', param2='scale', metric='auc')
            except Exception as e:
                print(f"Warning: Could not generate interaction plots. Reason: {e}")

        print("\n--- All Plot Generation Complete ---")
