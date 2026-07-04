"""Enhanced hyperparameter search with hierarchical optimization.

Integrates the HierarchicalSearchOptimizer with existing Grid/Random/Bayesian search.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import random

import numpy as np
import pandas as pd

# Import skopt types if available (used in hierarchical optimization)
try:
    from skopt.space import Real, Integer, Categorical
except ImportError:
    Real, Integer, Categorical = None, None, None


class HierarchicalHyperparameterSearch:
    """
    Orchestrates hierarchical hyperparameter search across multiple stages.

    Combines the flexibility of existing Grid/Random/Bayesian searches with
    intelligent search strategy optimization through:
    - Two-stage coarse-to-fine parameter search
    - Dynamic space reduction based on early results
    - Early stopping for unpromising trials
    - Parameter importance analysis to focus search

    This class acts as a wrapper that manages the search process at different
    levels of granularity, improving efficiency while maintaining or improving
    model quality.
    """

    def __init__(
        self,
        algorithm: Any,
        parameter_space: Dict[str, Any],
        method_name: str,
        global_params: Any,
        ml_grid_object: Any,
        max_total_evals: int = 100,
        reduction_factor: float = 0.4,
        eval_function: Optional[Callable] = None,
    ):
        """
        Initialize hierarchical hyperparameter search.

        Args:
            algorithm: The scikit-learn compatible estimator instance
            parameter_space: Hyperparameter search space dictionary
            method_name: Name of the algorithm for logging
            global_params: Global parameters singleton instance
            ml_grid_object: Main pipeline object containing data and settings
            max_total_evals: Total evaluation budget across all stages
            reduction_factor: Space reduction factor per stage (0-1)
            eval_function: Optional custom evaluation function
                Signature: params -> score, fit_time
        """
        self.algorithm = algorithm
        self.parameter_space = parameter_space
        self.method_name = method_name
        self.global_params = global_params
        self.ml_grid_object = ml_grid_object

        # hierarchical search configuration
        self.max_total_evals = max_total_evals
        self.reduction_factor = reduction_factor

        self.logger = logging.getLogger("ml_grid")

        # Use custom eval function or default to sklearn-based evaluation
        self.eval_function = eval_function

    def run_hierarchical_search(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, Dict[str, List[Dict]]]:
        """
        Execute hierarchical hyperparameter search.

        Args:
            X_train: Training features DataFrame
            y_train: Training labels Series

        Returns:
            Tuple of (best_estimator, all_results_by_stage)

        The method performs:
        1. Coarse search: Broad exploration with minimal evaluations per parameter
        2. Fine search: Focused exploitation on promising regions
        3. Refinement: Detailed optimization of top candidates

        Each stage uses dynamic space reduction based on previous results.
        """
        # Preprocess data for consistent format across stages
        X_train_preprocessed, y_train_preprocessed = self._preprocess_data(
            X_train, y_train
        )

        # Build evaluation function for search trials
        eval_fn = self._build_evaluation_function(
            X_train_preprocessed, y_train_preprocessed
        )

        # Import hierarchical optimizer
        from ml_grid.util.hierarchical_search import (
            HierarchicalSearchOptimizer,
            EarlyStoppingRule,
            DynamicSpaceReducer,
            ParameterImportanceAnalyzer,
        )

        # Initialize hierarchical search components
        self._space_optimizer = HierarchicalSearchOptimizer(
            initial_param_space=self.parameter_space,
            max_total_trials=self.max_total_evals,
            logger=self.logger,
        )

        _early_stopping = EarlyStoppingRule(min_trials=5, patience=10)
        space_reducer = DynamicSpaceReducer(self.parameter_space, self.logger)
        importance_analyzer = ParameterImportanceAnalyzer(self.logger)

        # Stage 1: Coarse Search - Broad exploration
        self.logger.info(f"Stage 1/3: Coarse search for {self.method_name}")

        # Initial coarse search - sample broadly
        coarse_results = self._run_search_stage(
            eval_fn,
            self.parameter_space,
            trial_budget=int(self.max_total_evals * 0.25),
            stage="coarse",
        )

        # Analyze parameter importance from coarse results
        importance_scores = importance_analyzer.analyze_parameters(
            coarse_results, self.parameter_space
        )

        self.logger.info(
            f"Stage 1 complete. Top params: {list(importance_scores.keys())[:5]}"
        )

        # Stage 2: Fine Search - Focus on important parameters
        self.logger.info(f"Stage 2/3: Fine search for {self.method_name}")

        # Reduce space based on coarse results and importance
        reduced_space = space_reducer.get_reduced_space(
            coarse_results, reduction_factor=0.4
        )

        fine_results = self._run_search_stage(
            eval_fn,
            reduced_space,
            trial_budget=int(self.max_total_evals * 0.45),
            stage="fine",
            importance_weights=importance_scores,
        )

        # Stage 3: Refinement - Detailed optimization
        self.logger.info(f"Stage 3/3: Refinement search for {self.method_name}")

        # Further reduce space to top candidates
        refined_space = space_reducer.get_reduced_space(
            fine_results, reduction_factor=0.25
        )

        refinement_results = self._run_search_stage(
            eval_fn,
            refined_space,
            trial_budget=int(self.max_total_evals * 0.30),
            stage="refinement",
            importance_weights=importance_scores,
        )

        # Combine all results and find best
        all_results = {
            "coarse": coarse_results,
            "fine": fine_results,
            "refinement": refinement_results,
        }

        combined_results = coarse_results + fine_results + refinement_results

        if not combined_results:
            self.logger.warning("No successful trials in hierarchical search")
            return None, all_results

        # Find best result
        best_result = max(combined_results, key=lambda x: x["score"])

        # Train final model with best parameters
        try:

            best_params = best_result["parameters"]

            if isinstance(best_params, dict):
                self.algorithm.set_params(**best_params)

            # Final training on full dataset
            X_train_array = X_train_preprocessed.to_numpy(dtype=float)
            y_train_array = y_train_preprocessed

            self.algorithm.fit(X_train_array, y_train_array)
            best_estimator = self.algorithm

        except Exception as e:
            self.logger.error(f"Failed to train final model: {e}")
            # Return None if final training fails
            best_estimator = None

        return best_estimator, all_results

    def _preprocess_data(self, X: pd.DataFrame, y) -> Tuple[pd.DataFrame, Any]:
        """Preprocess data for search evaluation."""
        # Copy and ensure consistent format
        X_copy = X.copy()

        # Ensure column names are strings (some models require this)
        X_copy.columns = X_copy.columns.astype(str)

        return X_copy, y

    def _build_evaluation_function(
        self, X_train: pd.DataFrame, y_train
    ) -> Callable[[Dict[str, Any]], Tuple[float, float]]:
        """
        Build evaluation function for search trials.

        Returns:
            Function that takes parameter dict and returns (score, fit_time)
        """

        def eval_fn(params: Dict[str, Any]) -> Tuple[float, float]:
            """Evaluate a single parameter combination."""
            import time
            from sklearn.model_selection import cross_val_score

            # Create copy of algorithm with new parameters
            try:
                model = self._clone_algorithm_with_params(params)

                start_time = time.time()

                # Perform quick CV evaluation (2 folds for speed)
                if hasattr(self.global_params, "metric_list"):
                    scoring = list(self.global_params.metric_list.keys())[0]
                else:
                    scoring = "roc_auc"

                scores = cross_val_score(
                    model,
                    X_train.to_numpy(dtype=float),
                    y_train,
                    cv=2,
                    scoring=scoring if isinstance(scoring, str) else None,
                    n_jobs=1,  # Single job for stability
                )

                fit_time = time.time() - start_time

                # Use mean score as metric
                score = scores.mean()

                return score, fit_time

            except Exception as e:
                self.logger.error(f"Evaluation error: {e}")
                return 0.0, 0.0

        return eval_fn

    def _clone_algorithm_with_params(self, params: Dict[str, Any]) -> Any:
        """Clone algorithm with specified parameters."""
        try:
            # Try to create new instance
            if hasattr(self.algorithm, "__class__"):
                new_model = self.algorithm.__class__()

                # Set parameters
                if isinstance(params, dict):
                    new_model.set_params(**params)

                return new_model
            else:
                # Fallback: use the same algorithm instance
                return self.algorithm

        except Exception:
            return self.algorithm

    def _run_search_stage(
        self,
        eval_fn: Callable[[Dict[str, Any]], Tuple[float, float]],
        param_space: Dict[str, Any],
        trial_budget: int,
        stage: str = "coarse",
        importance_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Run a single search stage.

        Args:
            eval_fn: Evaluation function for parameter combinations
            param_space: Parameter space for this stage
            trial_budget: Number of trials to run
            stage: Stage name for logging
            importance_weights: Optional parameter importance weights

        Returns:
            List of evaluation results with parameters and scores
        """

        results = []

        # Generate parameter combinations to evaluate
        param_combinations = self._generate_param_samples(
            param_space, trial_budget, importance_weights
        )

        for i, params in enumerate(param_combinations):
            try:
                score, fit_time = eval_fn(params)

                result = {
                    "parameters": params,
                    "score": score,
                    "fit_time": fit_time,
                    "trial_number": len(results) + 1,
                    "stage": stage,
                }

                results.append(result)

            except Exception as e:
                self.logger.debug(f"Trial {i+1} failed: {e}")

        return results

    def _generate_param_samples(
        self,
        param_space: Dict[str, Any],
        n_samples: int,
        importance_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate parameter samples for evaluation."""

        # Import skopt spaces if available
        from skopt.space import Real, Integer, Categorical

        samples = []
        has_skopt_spaces = False

        try:
            from skopt.space import Real, Integer, Categorical

            has_skopt_spaces = True
        except ImportError:
            pass

        if not has_skopt_spaces:
            # Skip skopt-specific sampling
            for _ in range(n_samples):
                sample = {
                    k: v[0] if isinstance(v, list) and len(v) > 0 else v
                    for k, v in param_space.items()
                }
                samples.append(sample)
            return samples

        for _ in range(n_samples):
            sample = {}

            for param_name, param_spec in param_space.items():
                # Determine if this is a skopt space
                is_skopt = False

                if isinstance(param_spec, (Real, Integer, Categorical)):
                    is_skopt = True
                elif isinstance(param_spec, (list, np.ndarray)):
                    # Check contents for skopt types
                    sample_values = (
                        list(param_spec)
                        if isinstance(param_spec, np.ndarray)
                        else param_spec
                    )
                    if any(
                        isinstance(v, (Real, Integer, Categorical))
                        for v in sample_values
                    ):
                        is_skopt = True

                if is_skopt and Real is not None:
                    # Sample from skopt space
                    sample[param_name] = self._sample_skopt_param(param_spec)
                elif isinstance(param_spec, list):
                    # List-based sampling with importance bias
                    values = param_spec

                    if importance_weights and param_name in importance_weights:
                        weight = importance_weights[param_name]

                        # Bias towards middle values with higher weight
                        mid_idx = len(values) // 2

                        if weight > 0.5:
                            # Narrow range around middle
                            narrowed_start = max(
                                0, mid_idx - int(len(values) * (1 - weight))
                            )
                            narrowed_end = min(
                                len(values),
                                mid_idx + int(len(values) * (1 - weight)) + 1,
                            )
                            values = param_spec[narrowed_start:narrowed_end]

                    sample[param_name] = random.choice(values)
                else:
                    # Scalar or other type
                    sample[param_name] = param_spec

            samples.append(sample)

        return samples

    def _sample_skopt_param(self, param_spec: Any) -> Union[int, float]:
        """Sample value from skopt parameter space."""

        if isinstance(param_spec, Real):
            # Sample based on prior distribution
            if hasattr(param_spec, "prior") and param_spec.prior == "log-uniform":
                return np.exp(
                    np.random.uniform(np.log(param_spec.low), np.log(param_spec.high))
                )
            else:
                return np.random.uniform(param_spec.low, param_spec.high)

        elif isinstance(param_spec, Integer):
            return np.random.randint(param_spec.low, param_spec.high + 1)

        elif isinstance(param_spec, Categorical):
            # Sample uniformly from categories
            weights = [1.0] * len(param_spec.categories)
            normalized_weights = [w / sum(weights) for w in weights]

            return np.random.choice(param_spec.categories, p=normalized_weights)

        else:
            # Fallback: sample from range if possible
            try:
                mid = (param_spec.high + param_spec.low) / 2
                span = (param_spec.high - param_spec.low) / 4
                return np.random.uniform(mid - span, mid + span)
            except Exception:
                return param_spec.low
