# plot_master.py
"""
Master plotting module that provides a single entry point to generate a
comprehensive set of visualizations for ML results analysis.
"""

import os
import pandas as pd
from typing import List, Optional, NoReturn
import logging

# Import all the individual plotter classes
from ml_grid.results_processing.plot_algorithms import AlgorithmComparisonPlotter
from ml_grid.results_processing.plot_distributions import DistributionPlotter
from ml_grid.results_processing.plot_features import FeatureAnalysisPlotter
from ml_grid.results_processing.plot_timeline import TimelineAnalysisPlotter
from ml_grid.results_processing.plot_hyperparameters import HyperparameterAnalysisPlotter
from ml_grid.results_processing.plot_feature_categories import FeatureCategoryPlotter
from ml_grid.results_processing.plot_pipeline_parameters import PipelineParameterPlotter
from ml_grid.results_processing.plot_global_importance import GlobalImportancePlotter
from ml_grid.results_processing.plot_interactions import InteractionPlotter
from ml_grid.results_processing.plot_best_model import BestModelAnalyzerPlotter
from ml_grid.results_processing.summarize_results import ResultsSummarizer


class MasterPlotter:
    """
    A facade that orchestrates specialized plotters to generate analysis plots.
    """

    def __init__(self, data: pd.DataFrame, output_dir: str = '.'):
        """Initializes the MasterPlotter with aggregated results data.

        This class acts as a facade, instantiating various specialized plotters
        to generate a comprehensive suite of analysis visualizations from the
        provided results DataFrame.

        Args:
            data (pd.DataFrame): A DataFrame containing the aggregated ML
                experiment results. Must be non-empty.
            output_dir (str, optional): The directory where output files (like
                CSVs) will be saved. Defaults to '.'.

        Raises:
            ValueError: If the input `data` is not a valid, non-empty
                pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")

        self.data = data
        self.output_dir = output_dir
        self.logger = logging.getLogger('ml_grid')

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
            self.logger.warning(f"Could not initialize GlobalImportancePlotter. Reason: {e}")

        # Pipeline parameter plotter
        try:
            self.pipeline_plotter = PipelineParameterPlotter(self.data)
        except ValueError as e:
            self.pipeline_plotter = None
            self.logger.warning(f"Could not initialize PipelineParameterPlotter. Reason: {e}")

        # Feature category plotter
        try:
            self.feature_cat_plotter = FeatureCategoryPlotter(self.data)
        except ValueError as e:
            self.feature_cat_plotter = None
            self.logger.warning(f"Could not initialize FeatureCategoryPlotter. Reason: {e}")

        # Feature plotter requires 'decoded_features' column
        if 'decoded_features' in self.data.columns:
            self.feature_plotter = FeatureAnalysisPlotter(self.data)
        else:
            self.feature_plotter = None
            self.logger.warning("'decoded_features' column not found. Feature-related plots will be skipped.")

        # Hyperparameter plotter requires 'algorithm_implementation' column
        if 'algorithm_implementation' in self.data.columns:
            self.hyperparam_plotter = HyperparameterAnalysisPlotter(self.data)
        else:
            self.hyperparam_plotter = None
            self.logger.warning("'algorithm_implementation' column not found. Hyperparameter-related plots will be skipped.")

        # Best model plotter
        try:
            self.best_model_plotter = BestModelAnalyzerPlotter(self.data)
        except ValueError as e:
            self.best_model_plotter = None
            self.logger.warning(f"Could not initialize BestModelAnalyzerPlotter. Reason: {e}")
            
        # Results summarizer
        try:
            self.summarizer = ResultsSummarizer(self.data)
        except ValueError as e:
            self.summarizer = None
            self.logger.warning(f"Could not initialize ResultsSummarizer. Reason: {e}")

    def plot_all(self,
                 metric: str = 'auc_m',
                 stratify_by_outcome: bool = True,
                 top_n_features: int = 20,
                 top_n_algorithms: int = 10,
                 save_best_results: bool = True) -> None:
        """Generates a comprehensive set of standard plots from all plotters.

        This method calls the main plotting functions from each specialized
        plotter to provide a full overview of the results, including algorithm
        comparisons, metric distributions, timeline trends, and feature
        importance. It also handles saving a summary of the best models.

        Args:
            metric (str, optional): The primary performance metric to use for
                plotting (e.g., 'auc', 'f1'). Defaults to 'auc_m'.
            stratify_by_outcome (bool, optional): If True, creates plots
                stratified by the 'outcome_variable' column. Defaults to True.
            top_n_features (int, optional): The number of top features to show
                in feature-related plots. Defaults to 20.
            top_n_algorithms (int, optional): The number of top algorithms to
                show in ranking plots. Defaults to 10.
            save_best_results (bool, optional): If True, saves a CSV summary of
                the best model per outcome. Defaults to True.
        """
        self.logger.info(f"--- Starting MasterPlotter.plot_all() ---")
        self.logger.info(f"Parameters: metric='{metric}', stratify_by_outcome={stratify_by_outcome}, save_best_results={save_best_results}")

        # --- Step 1: Generate and Save Summary Table First ---
        # This is done first to avoid being blocked by interactive plot windows.
        if self.summarizer and save_best_results:
            self.logger.info("\n>>> 1. Generating and Saving Best Model Summary Table...")
            try:
                # Use 'auc' as the default metric for the summary table, as it's most standard
                summary_metric = 'auc' if 'auc' in self.data.columns else metric
                self.logger.info(f"   - Using metric '{summary_metric}' for summary table.")

                if 'outcome_variable' not in self.data.columns:
                    raise ValueError("'outcome_variable' column is required to find the best model per outcome.") # This will be caught and logged
                if 'decoded_features' not in self.data.columns:
                     raise ValueError("'decoded_features' column is required to expand feature names.")

                best_models_df = self.summarizer.get_best_model_per_outcome(metric=summary_metric)

                # Ensure the output directory exists before saving
                os.makedirs(self.output_dir, exist_ok=True)

                output_path = os.path.join(self.output_dir, "best_models_summary.csv")
                best_models_df.to_csv(output_path, index=False)
                self.logger.info(f"✅ Best models summary successfully saved to: {os.path.abspath(output_path)}")
                
            except Exception as e:
                self.logger.error(f"❌ Could not generate or save best models summary table. Reason: {e}")
        elif not self.summarizer:
            self.logger.info("\n>>> 1. Skipping Best Model Summary Table: ResultsSummarizer was not initialized.")
        elif not save_best_results:
            self.logger.info("\n>>> 1. Skipping Best Model Summary Table: 'save_best_results' is False.")

        # --- Step 2: Generate Plots ---
        self.logger.info("\n>>> 2. Generating Algorithm Comparison Plots...")
        try:
            self.algo_plotter.plot_algorithm_boxplots(metric=metric, stratify_by_outcome=stratify_by_outcome)
            self.algo_plotter.plot_algorithm_performance_heatmap(metric=metric, aggregation='mean')
            self.algo_plotter.plot_algorithm_ranking(metric=metric, stratify_by_outcome=stratify_by_outcome, top_n=top_n_algorithms)
            self.algo_plotter.plot_algorithm_stability(metric=metric, top_n=top_n_algorithms)
            self.algo_plotter.plot_performance_tradeoff(metric_y=metric, metric_x='run_time', top_n_algos=top_n_algorithms)
            self.algo_plotter.plot_pareto_front(metric_y=metric, metric_x='run_time')
            self.algo_plotter.plot_statistical_significance_heatmap(metric=metric)
        except Exception as e:
            self.logger.warning(f"Could not generate algorithm plots. Reason: {e}")

        self.logger.info("\n>>> 3. Generating Distribution Plots...")
        try:
            self.dist_plotter.plot_metric_distributions(metrics=[metric, 'f1', 'mcc'], stratify_by_outcome=stratify_by_outcome)
            self.dist_plotter.plot_comparative_distributions(metric=metric, plot_type='violin')
        except Exception as e:
            self.logger.warning(f"Could not generate distribution plots. Reason: {e}")

        self.logger.info("\n>>> 4. Generating Timeline Plots...")
        try:
            self.timeline_plotter.plot_performance_timeline(metric=metric, stratify_by_outcome=stratify_by_outcome)
            self.timeline_plotter.plot_improvement_trends(metric=metric, stratify_by_outcome=stratify_by_outcome)
            self.timeline_plotter.plot_computational_cost_timeline(stratify_by_outcome=stratify_by_outcome)
        except Exception as e:
            self.logger.warning(f"Could not generate timeline plots. Reason: {e}")

        if self.feature_plotter:
            self.logger.info("\n>>> 5. Generating Feature Analysis Plots...")
            try:
                self.feature_plotter.plot_feature_usage_frequency(top_n=top_n_features, stratify_by_outcome=stratify_by_outcome)
                self.feature_plotter.plot_feature_performance_impact(metric=metric, top_n=top_n_features // 2)
                self.feature_plotter.plot_feature_metric_correlation(metric=metric, top_n=top_n_features // 2)
                self.feature_plotter.plot_feature_set_intersections(top_n_sets=15, stratify_by_outcome=stratify_by_outcome)
            except Exception as e:
                self.logger.warning(f"Could not generate feature plots. Reason: {e}")

        if self.hyperparam_plotter:
            self.logger.info("\n>>> 6. Generating Hyperparameter Analysis Plots...")
            try:
                # Get algorithms that have parsable hyperparameters
                available_algos = self.hyperparam_plotter.get_available_algorithms()
                if not available_algos:
                    self.logger.info("No algorithms with parsable hyperparameters found. Skipping hyperparameter plots.")
                else:
                    # Find the top overall algorithm *that has parsable hyperparameters*
                    plottable_data = self.data[self.data['method_name'].isin(available_algos)]

                    if plottable_data.empty:
                        self.logger.info("No data found for algorithms with parsable hyperparameters. Skipping.")
                    else:
                        top_algo = plottable_data.groupby('method_name')[metric].mean().idxmax()
                        self.logger.info(f"Analyzing hyperparameters for top algorithm: {top_algo}")

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
                                self.logger.info(f"No hyperparameters with multiple values found for {top_algo} to generate performance plot.")
            except Exception as e:
                self.logger.warning(f"Could not generate hyperparameter plots. Reason: {e}")

        if self.feature_cat_plotter:
            self.logger.info("\n>>> 7. Generating Feature Category Analysis Plots...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                category_metric = 'auc' if 'auc' in self.data.columns else metric
                self.logger.info(f"Analyzing feature category impact on metric: {category_metric.upper()}")
                self.feature_cat_plotter.plot_category_performance_boxplots(metric=category_metric)
                self.feature_cat_plotter.plot_category_impact_on_metric(metric=category_metric)
            except Exception as e:
                self.logger.warning(f"Could not generate feature category plots. Reason: {e}")

        if self.pipeline_plotter:
            self.logger.info("\n>>> 8. Generating Pipeline Parameter Analysis Plots...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                pipeline_metric = 'auc' if 'auc' in self.data.columns else metric
                self.logger.info(f"Analyzing pipeline parameter impact on metric: {pipeline_metric.upper()}")
                self.pipeline_plotter.plot_categorical_parameters(metric=pipeline_metric)
                self.pipeline_plotter.plot_continuous_parameters(metric=pipeline_metric)
            except Exception as e:
                self.logger.warning(f"Could not generate pipeline parameter plots. Reason: {e}")

        if self.global_importance_plotter:
            self.logger.info("\n>>> 9. Generating Global Importance Analysis Plot...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                global_metric = 'auc' if 'auc' in self.data.columns else metric
                self.global_importance_plotter.plot_global_importance(metric=global_metric, top_n=40)
            except Exception as e:
                self.logger.warning(f"Could not generate global importance plot. Reason: {e}")

        if self.interaction_plotter:
            self.logger.info("\n>>> 10. Generating Interaction Analysis Plots...")
            try:
                # Example interaction plot
                self.interaction_plotter.plot_categorical_interaction(param1='resample', param2='scale', metric='auc')
            except Exception as e:
                self.logger.warning(f"Could not generate interaction plots. Reason: {e}")

        if self.best_model_plotter:
            self.logger.info("\n>>> 11. Generating Best Model Analysis Plots...")
            try:
                # Use 'auc' as requested, but fall back to the main metric if 'auc' is not present
                best_model_metric = 'auc' if 'auc' in self.data.columns else metric
                self.best_model_plotter.plot_best_model_summary(metric=best_model_metric)
            except Exception as e:
                self.logger.warning(f"Could not generate best model plots. Reason: {e}")

        self.logger.info("\n--- All Plot Generation Complete ---")
