from typing import Any, Dict, List, Union

from skopt.space import Real, Integer, Categorical


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
            print(
                "Warning: Number of parameter combinations is not a positive integer. Returning 1."
            )
            print("combinations:", combinations)
            print("parameter_space:", single_space)
            print("steps:", steps)
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
