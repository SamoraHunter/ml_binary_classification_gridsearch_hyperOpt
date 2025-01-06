from itertools import product
import numpy as np
from skopt.space import Real, Integer, Categorical

def calculate_combinations(parameter_space, steps=10):
    """
    Approximates the number of parameter combinations for Bayesian optimization.
    Supports both a single dictionary or a list of dictionaries.
    
    Args:
        parameter_space (dict or list of dict): Either a single dictionary or a list of dictionaries, where each dictionary
                                                represents a parameter space. Keys are parameter names, and values are 
                                                skopt space objects like Real, Integer, or Categorical.
        steps (int): The granularity for continuous parameters (Real or Integer).
    
    Returns:
        int: Approximate number of parameter combinations across the provided parameter space(s).
    """
    def calculate_param_combinations(single_space, steps):
        combinations = 1
        for param, values in single_space.items():
            if isinstance(values, Real):
                combinations *= steps  # Steps define granularity for continuous values
            elif isinstance(values, Integer):
                combinations *= values.high - values.low + 1
            elif isinstance(values, Categorical):
                combinations *= len(values.categories)
            elif isinstance(values, list):
                combinations *= len(values)
        
        if not isinstance(combinations, (int, float)) or combinations <= 0:
            print("Warning: Number of parameter combinations is not an integer or <= 0. Returning 1.")
            print("combinations:", combinations)
            print("parameter_space:", single_space)
            print("steps:", steps)
            return 1
        
        return combinations
    
    if isinstance(parameter_space, dict):
        # Single parameter space
        return calculate_param_combinations(parameter_space, steps)
    elif isinstance(parameter_space, list):
        # List of parameter spaces
        return sum(calculate_param_combinations(space, steps) for space in parameter_space)
    else:
        raise ValueError("Invalid input: parameter_space must be a dict or a list of dicts.")
