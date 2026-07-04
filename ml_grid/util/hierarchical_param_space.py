"""Enhanced parameter space definitions with hierarchical search strategy.

This module provides advanced parameter space generation with:
- Two-stage coarse-to-fine parameter search
- Dynamic space reduction based on early results
- Early stopping for unpromising trials
- Parameter importance analysis to focus search

Key Classes:
    HierarchicalParamSpace: Extends ParamSpace with hierarchical search capabilities
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats
import logging
import threading


class HierarchicalParamSpace:
    """
    Implements hierarchical hyperparameter search strategy with the following stages:

    Stage 1 (Coarse): Broad exploration with minimal evaluations per parameter
    Stage 2 (Fine): Focused exploitation on promising regions
    Stage 3 (Refinement): Detailed optimization of top candidates

    Each stage reduces the search space based on previous results and parameter
    importance analysis.
    """

    def __init__(
        self,
        size: str = "medium",
        hierarchical_config: Dict[str, Any] = None,
        logger: logging.Logger = None,
    ):
        """
        Initialize hierarchical parameter space.

        Args:
            size: Base parameter space size ("medium", "xsmall", "xwide")
            hierarchical_config: Configuration for hierarchical search strategy
                - max_total_evals: Total evaluations budget (default: 100)
                - coarse_ratio: Proportion of trials for stage 1 (default: 0.25)
                - fine_ratio: Proportion of trials for stage 2 (default: 0.45)
                - refinement_ratio: Proportion of trials for stage 3 (default: 0.30)
                - reduction_factor: Space reduction factor per stage (default: 0.4)
            logger: Logger instance
        """
        self.base_size = size
        self.logger = logger or logging.getLogger("ml_grid")

        # Default hierarchical search configuration
        default_config = {
            "max_total_evals": 100,
            "coarse_ratio": 0.25,
            "fine_ratio": 0.45,
            "refinement_ratio": 0.30,
            "reduction_factor": 0.4,  # Keep 40% of space per stage
        }

        if hierarchical_config:
            default_config.update(hierarchical_config)

        self.hierarchical_config = default_config

        # Global parameters reference for bayessearch setting
        from ml_grid.util.global_params import global_parameters

        self._global_params = global_parameters

        self.logger.info(
            f"Hierarchical search initialized: "
            f"{default_config['max_total_evals']} total evaluations, "
            f"stages: {[int(default_config['max_total_evals'] * r) for r in [default_config['coarse_ratio'], default_config['fine_ratio'], default_config['refinement_ratio']]]}"
        )

    def generate_hierarchical_space(
        self, base_param_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Generate three-tiered parameter spaces for hierarchical search.

        Args:
            base_param_dict: Original parameter dictionary from ParamSpace

        Returns:
            Tuple of (coarse_space, fine_space, refinement_space)
        """
        return (
            self._reduce_space(base_param_dict, 0.5),  # Coarse: 50% of space
            self._reduce_space(base_param_dict, 0.3),  # Fine: 30% of space
            self._reduce_space(base_param_dict, 0.15),  # Refinement: 15% of space
        )

    def _reduce_space(
        self, param_dict: Dict[str, Any], reduction_factor: float
    ) -> Dict[str, Any]:
        """Reduce parameter space by selecting subset of values."""
        reduced = {}

        for param_name, param_values in param_dict.items():
            if isinstance(param_values, (list, np.ndarray)):
                # Convert to list if needed
                values_list = (
                    list(param_values)
                    if isinstance(param_values, np.ndarray)
                    else param_values
                )

                # Calculate reduced size
                reduced_size = max(1, int(len(values_list) * reduction_factor))

                # Select middle values for better coverage
                start_idx = (len(values_list) - reduced_size) // 2
                end_idx = start_idx + reduced_size

                reduced[param_name] = values_list[start_idx:end_idx]

            elif hasattr(param_values, "low") and hasattr(param_values, "high"):
                # skopt space types
                if param_values.low == param_values.high:
                    reduced[param_name] = param_values
                else:
                    span = param_values.high - param_values.low
                    center = (param_values.low + param_values.high) / 2
                    new_span = span * reduction_factor

                    if hasattr(param_values, "prior"):
                        # Real space
                        from skopt.space import Real

                        reduced[param_name] = Real(
                            max(0, center - new_span / 2),
                            center + new_span / 2,
                            prior=param_values.prior,
                        )
                    else:
                        # Integer space
                        from skopt.space import Integer

                        reduced[param_name] = Integer(
                            int(max(0, center - new_span / 2)),
                            int(center + new_span / 2),
                        )
            else:
                reduced[param_name] = param_values

        return reduced


class AdaptiveParameterAnalyzer:
    """Analyzes parameter importance and guides search focus."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("ml_grid")
        self._lock = threading.Lock()

    def analyze_parameter_importance(
        self, results_list: List[Dict[str, Any]], param_space: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze which parameters most impact model performance.

        Uses statistical methods to identify important parameters:
        - ANOVA for categorical parameters
        - Spearman correlation for continuous parameters

        Args:
            results_list: List of result dictionaries with 'score' and parameter values
            param_space: Original parameter space

        Returns:
            Dictionary mapping parameter names to importance scores (0-1)
        """
        if len(results_list) < 5:
            self.logger.debug("Insufficient results for importance analysis")
            return {name: 1.0 for name in param_space.keys()}

        import pandas as pd

        # Convert results to DataFrame
        df_data = []
        for result in results_list:
            row = {"score": result.get("score", 0)}

            params = result.get("parameters", {})
            if isinstance(params, dict):
                row.update(params)

            df_data.append(row)

        df = pd.DataFrame(df_data)

        importance_scores = {}

        with self._lock:
            for param_name in param_space.keys():
                if param_name not in df.columns or "score" not in df.columns:
                    importance_scores[param_name] = 1.0
                    continue

                try:
                    param_values = df[param_name].dropna()
                    scores = df["score"]

                    # Check parameter type and apply appropriate analysis
                    unique_count = param_values.nunique()

                    if unique_count <= 5:
                        # Categorical/discrete analysis using ANOVA
                        groups = []
                        group_names = []

                        for val in param_values.unique():
                            subset_scores = scores[param_values == val]
                            if len(subset_scores) >= 2:
                                groups.append(subset_scores.values)
                                group_names.append(str(val))

                        if len(groups) >= 2:
                            f_stat, p_val = stats.f_oneway(*groups)
                            # Normalized importance based on F-statistic
                            importance = min(
                                1.0, max(0.1, (f_stat / (f_stat + len(groups))))
                            )
                            importance_scores[param_name] = importance
                        else:
                            importance_scores[param_name] = 1.0
                    else:
                        # Continuous parameter analysis using correlation
                        if (
                            pd.api.types.is_numeric_dtype(param_values)
                            and unique_count > 1
                        ):
                            corr, p_val = stats.spearmanr(
                                param_values.values.astype(float),
                                scores.values.astype(float),
                            )
                            importance_scores[param_name] = max(0.1, abs(corr))
                        else:
                            # Non-numeric but multiple values - use variance
                            variance_scores = [
                                scores.values.std() for _ in param_values.unique()
                            ]
                            importance_scores[param_name] = max(
                                0.1, min(1.0, len(variance_scores) / len(df))
                            )

                except Exception as e:
                    self.logger.debug(f"Error analyzing {param_name}: {e}")
                    importance_scores[param_name] = 1.0

        # Normalize scores
        if importance_scores:
            max_score = max(importance_scores.values())
            if max_score > 0:
                importance_scores = {
                    k: v / max_score for k, v in importance_scores.items()
                }

        self.logger.debug(f"Parameter importance: {importance_scores}")

        return importance_scores

    def get_early_stopping_rule(
        self, min_trials: int = 5, patience: int = 10
    ) -> Dict[str, Any]:
        """
        Create early stopping configuration.

        Args:
            min_trials: Minimum trials before checking early stopping
            patience: Number of consecutive non-improving trials

        Returns:
            Early stopping configuration dict
        """
        return {
            "enabled": True,
            "min_trials": min_trials,
            "patience": patience,
            "monitor_metric": "score",
            "mode": "max",  # Higher scores are better
        }


class HierarchicalSearchManager:
    """Manages the hierarchical search process."""

    def __init__(
        self,
        param_space: Dict[str, Any],
        max_total_evals: int = 100,
        reduction_factor: float = 0.4,
        logger: logging.Logger = None,
    ):
        self.param_space = param_space.copy()
        self.max_total_evals = max_total_evals
        self.reduction_factor = reduction_factor

        self.logger = logger or logging.getLogger("ml_grid")
        self.results_history: List[Dict[str, Any]] = []

        self._analyzer = AdaptiveParameterAnalyzer(logger)

    def get_staged_spaces(self, num_stages: int = 3) -> Dict[int, Dict[str, Any]]:
        """
        Generate parameter spaces for each hierarchical stage.

        Args:
            num_stages: Number of stages (default: 3 for coarse->fine->refine)

        Returns:
            Dictionary mapping stage number to reduced param space
        """
        staged_spaces = {}
        current_space = self.param_space.copy()

        # Allocation ratios for each stage
        stage_ratios = [0.25, 0.45, 0.30]  # Coarse, Fine, Refinement

        for stage in range(1, num_stages + 1):
            if stage > len(stage_ratios):
                break

            # Calculate reduction factor per stage
            # Each stage reduces the space by reduction_factor
            current_reduction = self.reduction_factor ** (
                (stage - 1) / (num_stages - 1)
            )

            # Get reduced space for this stage
            staged_spaces[stage] = self._reduce_space_for_stage(
                current_space, current_reduction
            )

            # For next iteration, reduce the current space further
            current_space = staged_spaces[stage]

        return staged_spaces

    def _reduce_space_for_stage(
        self, param_space: Dict[str, Any], reduction_factor: float
    ) -> Dict[str, Any]:
        """Reduce parameter space for a single stage."""
        reduced = {}

        for param_name, param_spec in param_space.items():
            if isinstance(param_spec, list):
                # List-based spaces - sample from middle range
                if len(param_spec) > 1:
                    mid_idx = len(param_spec) // 2
                    reduced_size = max(2, int(len(param_spec) * reduction_factor))

                    start_idx = max(0, mid_idx - reduced_size // 2)
                    end_idx = min(len(param_spec), start_idx + reduced_size)

                    reduced[param_name] = param_spec[start_idx:end_idx]
                else:
                    reduced[param_name] = param_spec

            elif isinstance(param_spec, dict):
                # Nested dictionaries (like data feature toggles)
                reduced[param_name] = self._reduce_nested_space(
                    param_spec, reduction_factor
                )

            elif hasattr(param_spec, "low") and hasattr(param_spec, "high"):
                # skopt space types - narrow the range
                span = param_spec.high - param_spec.low
                center = (param_spec.low + param_spec.high) / 2

                new_span = span * reduction_factor

                if hasattr(param_spec, "prior"):  # Real space
                    from skopt.space import Real

                    reduced[param_name] = Real(
                        max(0, center - new_span / 2),
                        center + new_span / 2,
                        prior=param_spec.prior,
                    )
                else:  # Integer space
                    from skopt.space import Integer

                    reduced[param_name] = Integer(
                        int(max(0, center - new_span / 2)), int(center + new_span / 2)
                    )
            else:
                reduced[param_name] = param_spec

        return reduced

    def _reduce_nested_space(
        self, nested_dict: Dict[str, List], reduction_factor: float
    ) -> Dict[str, List]:
        """Reduce nested feature selection space."""
        reduced = {}

        # Keep top features based on likely importance (first few typically more important)
        sorted_items = list(nested_dict.items())

        # For binary toggles, keep all but reduce the combinatorial space
        for key, value in sorted_items[
            : max(5, int(len(sorted_items) * reduction_factor))
        ]:
            if isinstance(value, list):
                mid_idx = len(value) // 2
                reduced[key] = [value[mid_idx]] if mid_idx < len(value) else value
            else:
                reduced[key] = value

        return reduced

    def update_with_results(self, trial_result: Dict[str, Any]) -> None:
        """Update history with new trial results."""
        self.results_history.append(trial_result)

    def should_early_stop(self) -> Tuple[bool, str]:
        """
        Check if search should stop based on early stopping rules.

        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        if len(self.results_history) < 5:
            return False, "Insufficient trials"

        # Get recent scores
        recent_scores = [r.get("score", 0) for r in self.results_history[-10:]]
        best_score = max(recent_scores)

        # Check for improvement in last N trials
        last_best = max(self.results_history[-5:].get("score", 0))

        if last_best >= best_score * 0.98:
            return True, "No significant improvement in recent trials"

        return False, "Continuing"


def create_hierarchical_param_space(
    size: str = "medium",
    max_total_evals: int = 100,
    reduction_factor: float = 0.4,
    logger: logging.Logger = None,
) -> HierarchicalParamSpace:
    """
    Factory function to create hierarchical parameter space.

    Args:
        size: Base parameter space size
        max_total_evals: Total evaluation budget
        reduction_factor: Space reduction per stage
        logger: Logger instance

    Returns:
        Configured HierarchicalParamSpace instance
    """
    config = {
        "max_total_evals": max_total_evals,
        "coarse_ratio": 0.25,
        "fine_ratio": 0.45,
        "refinement_ratio": 0.30,
        "reduction_factor": reduction_factor,
    }

    return HierarchicalParamSpace(size=size, hierarchical_config=config, logger=logger)
