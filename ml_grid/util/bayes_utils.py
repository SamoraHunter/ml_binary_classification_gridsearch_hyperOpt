import logging
from typing import Any, Dict, List, Union

from skopt.space import Categorical, Integer, Real


def calculate_combinations(
    parameter_space: Union[Dict[str, Any], List[Dict[str, Any]]], steps: int = 10
) -> int:
    """Approximates the number of parameter combinations for hyperparameter search.

    This function supports both a single dictionary or a list of dictionaries for
    the parameter space. It is useful for estimating the size of a search space,
    especially for Bayesian optimization where continuous parameters are sampled.

    Args:
        parameter_space (Union[Dict[str, Any], List[Dict[str, Any]]]): A single
            dictionary or a list of dictionaries representing the parameter
            space. Keys are parameter names, and values can be `skopt.space`
            objects (Real, Integer, Categorical) or simple lists.
        steps (int, optional): The granularity for discretizing continuous
            parameters (`Real`). Defaults to 10.

    Returns:
        int: The approximate total number of parameter combinations.

    Raises:
        ValueError: If `parameter_space` is not a dict or a list of dicts.
    """

    def calculate_param_combinations(single_space: Dict[str, Any], steps: int) -> int:
        """Calculates combinations for a single parameter space dictionary."""
        logger = logging.getLogger("ml_grid")
        combinations = 1
        for param, values in single_space.items():
            if isinstance(values, Real):
                combinations *= steps  # Steps define granularity for continuous values
            elif isinstance(values, Integer):
                combinations *= (values.high - values.low) + 1
            elif isinstance(values, Categorical):
                combinations *= len(values.categories)
            elif isinstance(values, list):
                combinations *= len(values)

        if not isinstance(combinations, (int, float)) or combinations <= 0:
            logger.warning(
                "Number of parameter combinations is not a positive integer. Returning 1."
            )
            logger.warning(f"Combinations calculated: {combinations}")
            logger.warning(f"Parameter space: {single_space}")
            return 1

        return int(combinations)

    if isinstance(parameter_space, dict):
        # Single parameter space
        return calculate_param_combinations(parameter_space, steps)
    elif isinstance(parameter_space, list):
        # List of parameter spaces
        return sum(
            calculate_param_combinations(space, steps) for space in parameter_space
        )
    else:
        raise ValueError(
            "Invalid input: parameter_space must be a dict or a list of dicts."
        )


def is_skopt_space(param_value: Any) -> bool:
    """Checks if a parameter value is a scikit-optimize space object.

    Args:
        param_value (Any): The parameter value to check.

    Returns:
        bool: True if the value is an instance of Real, Integer, or Categorical from skopt.
    """
    return isinstance(param_value, (Real, Integer, Categorical))
