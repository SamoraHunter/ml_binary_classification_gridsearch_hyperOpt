"""Hierarchical Hyperparameter Search Module.

This module implements advanced optimization strategies for hyperparameter search:
- Two-stage coarse-to-fine parameter search
- Dynamic space reduction based on early results
- Early stopping for unpromising trials
- Parameter importance analysis to focus search
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
import time
import threading
from dataclasses import dataclass

# Import sklearn ParameterGrid for backward compatibility
try:
    from sklearn.model_selection import ParameterGrid
except ImportError:
    # Fallback if sklearn is not available
    class ParameterGrid:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 0


from skopt.space import Real, Integer, Categorical
from scipy import stats

# ============================================================================
# Hierarchical Search Strategy
# ============================================================================


@dataclass
class SearchResult:
    """Stores results from a single hyperparameter evaluation."""

    parameters: Dict[str, Any]
    score: float
    fit_time: float = 0.0
    trial_number: int = 0
    stage: str = "coarse"
    confidence: float = 1.0


# Backward compatibility: Result is now the same as SearchResult
Result = SearchResult


class ParameterImportanceAnalyzer:
    """Analyzes parameter importance using statistical methods."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("ml_grid")
        self._lock = threading.Lock()

    def analyze_parameters(
        self, results: List[SearchResult], param_space: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze parameter importance based on cross-validation results.

        Uses correlation analysis to identify which parameters have the most
        significant impact on model performance.

        Args:
            results: List of search results with parameters and scores
            param_space: Original parameter space for reference

        Returns:
            Dictionary mapping parameter names to importance scores (0-1)
        """
        if len(results) < 3:
            self.logger.debug("Insufficient results for importance analysis")
            return {name: 1.0 for name in param_space.keys()}

        # Extract parameter values and scores
        df_data = []
        for result in results:
            row = {"score": result.score}
            params = result.parameters

            # Handle both dict and list[dict] parameter formats
            if isinstance(params, dict):
                for key, value in params.items():
                    row[key] = self._param_to_scalar(value)

            df_data.append(row)

        if len(df_data) < 2:
            return {name: 1.0 for name in param_space.keys()}

        df = pd.DataFrame(df_data)

        importance_scores = {}
        score_col = "score"

        with self._lock:
            for col in df.columns:
                if col == score_col:
                    continue

                try:
                    # Check parameter type
                    values = df[col].dropna()

                    if len(values) < 2:
                        importance_scores[col] = 1.0
                        continue

                    # Categorical or discrete parameters
                    unique_vals = values.nunique()
                    if unique_vals <= 5:
                        # Use ANOVA for categorical
                        groups = [
                            df[df[col] == v][score_col].values
                            for v in df[col].unique()
                            if len(df[df[col] == v]) > 0
                        ]

                        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                            f_stat, p_val = stats.f_oneway(*groups)
                            importance_scores[col] = min(
                                1.0, max(0.1, (f_stat / (f_stat + len(groups))))
                            )
                        else:
                            importance_scores[col] = 1.0
                    else:
                        # Continuous parameters - use correlation
                        if unique_vals > 1 and pd.api.types.is_numeric_dtype(values):
                            corr, p_val = stats.spearmanr(values, df[score_col])
                            # Convert to positive importance score
                            importance_scores[col] = max(0.1, abs(corr))
                        else:
                            importance_scores[col] = 1.0

                except Exception as e:
                    self.logger.debug(f"Error analyzing parameter {col}: {e}")
                    importance_scores[col] = 1.0

        # Normalize to 0-1 range
        if importance_scores:
            max_score = max(importance_scores.values())
            if max_score > 0:
                importance_scores = {
                    k: v / max_score for k, v in importance_scores.items()
                }

        self.logger.debug(f"Parameter importance analysis: {importance_scores}")
        return importance_scores

    def _param_to_scalar(self, value: Any) -> Union[int, float, str]:
        """Convert parameter value to scalar for analysis."""
        if isinstance(value, (bool, type(None))):
            return int(value) if value is not None else -1
        elif isinstance(value, (int, float)):
            return value
        elif hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        else:
            return hash(str(value)) % 1000


class DynamicSpaceReducer:
    """Dynamically reduces search space based on early results."""

    def __init__(self, initial_space: Dict[str, Any], logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("ml_grid")
        self.initial_space = initial_space.copy()
        self.top_n_percentile = 25  # Keep top 25% of parameter combinations

    def get_reduced_space(
        self, results: List[SearchResult], reduction_factor: float = 0.3
    ) -> Dict[str, Any]:
        """
        Reduce the search space based on top performing parameter values.

        Args:
            results: Results from previous search stage
            reduction_factor: Fraction of original space to retain (0-1)

        Returns:
            Reduced parameter space for next stage
        """
        if len(results) < 5:
            self.logger.debug("Insufficient data for space reduction")
            return self.initial_space

        # Sort by score and get top results
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        top_count = max(1, int(len(sorted_results) * reduction_factor))
        top_results = sorted_results[:top_count]

        reduced_space = {}

        for param_name, param_values in self.initial_space.items():
            # Extract all values seen for this parameter
            value_counts = {}
            for result in top_results:
                params = result.parameters

                if isinstance(params, dict) and param_name in params:
                    value = self._param_to_scalar_for_reduction(params[param_name])
                    value_counts[value] = value_counts.get(value, 0) + 1

            # If we saw this parameter in top results
            if value_counts:
                # Get most frequent values (mode)
                sorted_values = sorted(
                    value_counts.items(), key=lambda x: x[1], reverse=True
                )

                if len(sorted_values) > 0:
                    # Keep top N values with highest frequency
                    top_values = [
                        v[0] for v in sorted_values[: max(2, len(sorted_values) // 2)]
                    ]

                    # Convert back to proper type
                    reduced_space[param_name] = self._restore_param_type(
                        param_values, top_values
                    )

            # If no reduction info, keep original
            if param_name not in reduced_space:
                reduced_space[param_name] = param_values

        # Log space reduction
        original_size = (
            len(ParameterGrid(self.initial_space))
            if hasattr(self.initial_space, "values")
            else 0
        )
        reduced_size = (
            len(ParameterGrid(reduced_space)) if hasattr(reduced_space, "values") else 0
        )

        self.logger.info(
            f"Space reduced from ~{original_size} to ~{reduced_size} combinations"
        )

        return reduced_space

    def _param_to_scalar_for_reduction(self, value: Any) -> Union[int, float, str]:
        """Convert parameter for frequency analysis."""
        try:
            if hasattr(value, "item"):
                item = value.item()
                return int(item) if isinstance(item, (int, np.integer)) else float(item)
            elif isinstance(value, bool):
                return int(value)
            return hash(str(value))
        except Exception:
            return hash(str(value))

    def _restore_param_type(
        self, original_value: Any, new_values: List[Union[int, float]]
    ) -> Any:
        """Restore parameter type from reduced values."""
        if isinstance(original_value, list):
            # Preserve dtype for lists
            dtype = type(original_value[0]) if len(original_value) > 0 else float
            try:
                return [
                    dtype(v) if not isinstance(v, bool) else bool(v) for v in new_values
                ]
            except Exception:
                return new_values

        elif isinstance(original_value, (Real, Integer, Categorical)):
            # For skopt spaces, create new space with top values
            if hasattr(original_value, "prior"):
                # Real space
                domain = list(set(new_values))
                return Real(
                    min(domain), max(original_value.high), prior=original_value.prior
                )

        elif isinstance(original_value, dict):
            return {
                k: v
                for k, v in zip(
                    list(original_value.keys())[: len(new_values)], new_values
                )
            }

        return original_value


class EarlyStoppingRule:
    """Early stopping rules based on trial performance."""

    def __init__(
        self, min_trials: int = 5, patience: int = 10, threshold_factor: float = 0.95
    ):
        self.min_trials = min_trials
        self.patience = patience
        self.threshold_factor = threshold_factor

        self._best_score = float("-inf")
        self._no_improvement_count = 0

    def should_stop(self, results: List[SearchResult]) -> Tuple[bool, str]:
        """
        Determine if search should stop based on early stopping rules.

        Args:
            results: All results collected so far

        Returns:
            Tuple of (should_stop, reason)
        """
        if len(results) < self.min_trials:
            return False, "Insufficient trials"

        current_score = results[-1].score

        # Check for new best
        if current_score > self._best_score:
            self._best_score = current_score
            self._no_improvement_count = 0
            return False, f"New best: {current_score:.4f}"

        self._no_improvement_count += 1

        # Check patience
        if self._no_improvement_count >= self.patience:
            return True, f"No improvement for {self.patience} trials"

        return False, f"No improvement ({self._no_improvement_count}/{self.patience})"


class HierarchicalSearchOptimizer:
    """
    Implements hierarchical hyperparameter search with following stages:

    1. Coarse Search: Broad exploration with minimal evaluations per parameter
    2. Fine Search: Focused exploitation on promising regions
    3. Refinement: Detailed optimization of top candidates

    Each stage uses dynamic space reduction and early stopping.
    """

    def __init__(
        self,
        initial_param_space: Dict[str, Any],
        max_total_trials: int = 100,
        coarse_ratio: float = 0.25,
        fine_ratio: float = 0.45,
        refinement_ratio: float = 0.30,
        logger: logging.Logger = None,
    ):
        self.logger = logger or logging.getLogger("ml_grid")

        self.initial_space = initial_param_space.copy()
        self.max_total_trials = max_total_trials

        # Stage allocation ratios
        self.coarse_ratio = coarse_ratio
        self.fine_ratio = fine_ratio
        self.refinement_ratio = refinement_ratio

        # Results storage
        self.results: List[SearchResult] = []

        # Optimizers for each stage
        self._analyzer = ParameterImportanceAnalyzer(self.logger)
        self._space_reducer = DynamicSpaceReducer(initial_param_space, self.logger)
        self._early_stopping = EarlyStoppingRule()

    def run_hierarchical_search(
        self,
        evaluate_fn: callable,
        max_trials_per_stage: Optional[Dict[str, int]] = None,
        verbose: bool = True,
    ) -> Tuple[SearchResult, Dict[str, List[SearchResult]]]:
        """
        Execute hierarchical hyperparameter search.

        Args:
            evaluate_fn: Function to evaluate parameter set:
                params -> score, fit_time
            max_trials_per_stage: Optional override for trial counts per stage
            verbose: Whether to log progress

        Returns:
            Tuple of (best_result, all_results_by_stage)
        """
        total_budget = self.max_total_trials

        # Calculate trials per stage
        coarse_trials = int(total_budget * self.coarse_ratio)
        fine_trials = int(total_budget * self.fine_ratio)
        refinement_trials = total_budget - coarse_trials - fine_trials

        if max_trials_per_stage:
            coarse_trials = max_trials_per_stage.get("coarse", coarse_trials)
            fine_trials = max_trials_per_stage.get("fine", fine_trials)
            refinement_trials = max_trials_per_stage.get(
                "refinement", refinement_trials
            )

        all_results = {}
        current_space = self.initial_space

        # Stage 1: Coarse Search
        if verbose:
            self.logger.info(f"Stage 1/3: Coarse search ({coarse_trials} trials)")

        coarse_results = self._run_stage(
            "coarse",
            current_space,
            evaluate_fn,
            min_trials=coarse_trials,
            exploration_factor=2.0,
        )

        all_results["coarse"] = coarse_results

        # Analyze parameter importance from coarse results
        importance = self._analyzer.analyze_parameters(coarse_results, current_space)

        if verbose:
            self.logger.info(f"Parameter importance: {importance}")

        # Stage 2: Fine Search - focus on important parameters
        if verbose:
            self.logger.info(f"Stage 2/3: Fine search ({fine_trials} trials)")

        # Reduce space based on coarse results and importance
        coarse_space_reduced = self._space_reducer.get_reduced_space(
            coarse_results, reduction_factor=0.4
        )

        fine_results = self._run_stage(
            "fine",
            coarse_space_reduced,
            evaluate_fn,
            min_trials=fine_trials,
            exploration_factor=1.5,
            importance_weights=importance,
        )

        all_results["fine"] = fine_results

        # Stage 3: Refinement - exploit top candidates
        if verbose:
            self.logger.info(f"Stage 3/3: Refinement ({refinement_trials} trials)")

        refined_space = self._space_reducer.get_reduced_space(
            fine_results, reduction_factor=0.25
        )

        refinement_results = self._run_stage(
            "refinement",
            refined_space,
            evaluate_fn,
            min_trials=refinement_trials,
            exploration_factor=1.0,
            importance_weights=importance,
        )

        all_results["refinement"] = refinement_results

        # Combine and find best
        combined_results = coarse_results + fine_results + refinement_results
        best_result = max(combined_results, key=lambda x: x.score)

        if verbose:
            self.logger.info(
                f"Hierarchical search complete. Best score: {best_result.score:.4f}"
            )

        return best_result, all_results

    def _run_stage(
        self,
        stage_name: str,
        param_space: Dict[str, Any],
        evaluate_fn: callable,
        min_trials: int,
        exploration_factor: float = 1.0,
        importance_weights: Optional[Dict[str, float]] = None,
    ) -> List[SearchResult]:
        """Run a single search stage."""
        results = []

        # Calculate trials with exploration factor
        effective_trials = max(1, int(min_trials * exploration_factor))

        trial_number = len(self.results) + 1

        # Determine sampling strategy based on space type
        skopt_types = (Real, Integer, Categorical)
        has_skopt = any(
            isinstance(v, skopt_types)
            or (isinstance(v, list) and any(isinstance(x, skopt_types) for x in v))
            for v in param_space.values()
        )

        if has_skopt:
            # Use Bayesian sampling
            params_list = self._sample_with_bayesian(
                param_space, effective_trials, importance_weights
            )
        else:
            # Use grid/random sampling
            params_list = self._sample_grid(param_space, effective_trials)

        # Evaluate parameters
        for params in params_list:
            _start_time = time.time()

            try:
                score, fit_time = evaluate_fn(params)

                result = Result(
                    parameters=params,
                    score=score,
                    fit_time=fit_time,
                    trial_number=trial_number,
                    stage=stage_name,
                    confidence=self._calculate_confidence(stage_name, results),
                )

                self.results.append(result)
                results.append(result)

                trial_number += 1

            except Exception as e:
                self.logger.warning(f"Trial {trial_number} failed: {e}")

                result = Result(
                    parameters=params,
                    score=0.0,
                    fit_time=0.0,
                    trial_number=trial_number,
                    stage=stage_name,
                    confidence=0.1,
                )

                self.results.append(result)
                results.append(result)

                trial_number += 1

        # Check early stopping
        should_stop, reason = self._early_stopping.should_stop(results)

        if should_stop:
            self.logger.info(f"Early stopping in {stage_name} stage: {reason}")

        return results

    def _sample_grid(
        self, param_space: Dict[str, Any], n_samples: int
    ) -> List[Dict[str, Any]]:
        """Sample parameter combinations from grid."""
        try:
            param_grid = list(ParameterGrid(param_space))

            if len(param_grid) <= n_samples:
                return param_grid

            # Random sample without replacement
            rng = np.random.RandomState(42)
            indices = rng.choice(len(param_grid), size=n_samples, replace=False)
            return [param_grid[i] for i in indices]

        except Exception:
            # Fall back to random sampling if ParameterGrid fails
            return self._sample_random(param_space, n_samples)

    def _sample_with_bayesian(
        self,
        param_space: Dict[str, Any],
        n_samples: int,
        importance_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample parameters using importance-weighted Bayesian approach."""
        # For simplicity, use stratified sampling with importance weighting
        params_list = []

        for _ in range(n_samples):
            sample = {}

            for param_name, param_spec in param_space.items():
                if importance_weights is None or param_name not in importance_weights:
                    weight = 1.0
                else:
                    weight = importance_weights[param_name]

                # Bias towards middle values with higher weight parameters
                biased_sampling = self._sample_param(param_spec, bias=weight)
                sample[param_name] = biased_sampling

            params_list.append(sample)

        return params_list

    def _sample_random(
        self, param_space: Dict[str, Any], n_samples: int
    ) -> List[Dict[str, Any]]:
        """Random sampling for parameter combinations."""
        params_list = []

        for _ in range(n_samples):
            sample = {}

            for param_name, param_spec in param_space.items():
                if isinstance(param_spec, Real):
                    value = np.random.uniform(param_spec.low, param_spec.high)
                elif isinstance(param_spec, Integer):
                    value = np.random.randint(param_spec.low, param_spec.high + 1)
                elif isinstance(param_spec, Categorical):
                    value = np.random.choice(param_spec.categories)
                elif isinstance(param_spec, list):
                    value = np.random.choice(param_spec)
                else:
                    value = param_spec

                sample[param_name] = value

            params_list.append(sample)

        return params_list

    def _sample_param(
        self, param_spec: Any, bias: float = 1.0
    ) -> Union[int, float, Any]:
        """Sample a single parameter with optional bias."""
        if isinstance(param_spec, Real):
            # Bias towards middle with higher weight
            mid = (param_spec.high + param_spec.low) / 2
            spread = (param_spec.high - param_spec.low) / 2

            # Apply bias: higher weight pushes towards center
            center_bias = (1 - bias) * spread
            low = mid - center_bias
            high = mid + center_bias

            return np.random.uniform(low, high)

        elif isinstance(param_spec, Integer):
            mid = (param_spec.high + param_spec.low) // 2

            if bias > 0.5:
                # Bias towards center values
                range_size = max(1, int((param_spec.high - param_spec.low) * bias / 2))
                low = max(param_spec.low, mid - range_size)
                high = min(param_spec.high, mid + range_size)
            else:
                low, high = param_spec.low, param_spec.high

            return np.random.randint(low, high + 1)

        elif isinstance(param_spec, Categorical):
            categories = list(param_spec.categories)

            if len(categories) <= 2:
                return np.random.choice(categories)

            # Bias towards middle categories
            mid_idx = len(categories) // 2

            if bias > 0.5:
                # Narrow to middle categories
                start = max(0, mid_idx - int(len(categories) * (1 - bias)))
                end = min(
                    len(categories), mid_idx + int(len(categories) * (1 - bias)) + 1
                )
                return np.random.choice(categories[start:end])

            return np.random.choice(categories)

        elif isinstance(param_spec, list):
            if len(param_spec) == 0:
                return []

            if len(param_spec) <= 2:
                return np.random.choice(param_spec)

            # Bias towards middle values
            mid_idx = len(param_spec) // 2

            if bias > 0.5:
                start = max(0, mid_idx - int(len(param_spec) * (1 - bias)))
                end = min(
                    len(param_spec), mid_idx + int(len(param_spec) * (1 - bias)) + 1
                )
                return np.random.choice(param_spec[start:end])

            return np.random.choice(param_spec)

        return param_spec

    def _calculate_confidence(
        self, stage: str, current_results: List[SearchResult]
    ) -> float:
        """Calculate confidence score for a result."""
        # Increase confidence as we progress through stages
        stage_weights = {"coarse": 0.5, "fine": 0.75, "refinement": 1.0}

        base_confidence = stage_weights.get(stage, 0.5)

        # Adjust based on result count (more reliable with more data)
        if len(current_results) >= 20:
            base_confidence *= 1.2
        elif len(current_results) >= 10:
            base_confidence *= 1.1

        return min(1.0, base_confidence)


def optimize_hyperparameter_search(
    param_space: Dict[str, Any], max_total_evals: int = 100, verbose: bool = True
) -> HierarchicalSearchOptimizer:
    """
    Create and configure a hierarchical search optimizer.

    Args:
        param_space: Initial parameter space dictionary
        max_total_evals: Total number of evaluations to perform
        verbose: Whether to enable verbose logging

    Returns:
        Configured HierarchicalSearchOptimizer instance
    """
    return HierarchicalSearchOptimizer(
        initial_param_space=param_space,
        max_total_trials=max_total_evals,
        logger=logging.getLogger("ml_grid"),
    )
