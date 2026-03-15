"""Functions to validate model-specific hyperparameters before grid search."""

import logging
from typing import Any, Dict, List, Union

from sklearn.neighbors import KNeighborsClassifier

# from ml_grid.model_classes.knn_gpu_classifier_class import KNNGpuWrapperClass
# from ml_grid.model_classes.knn_wrapper_class import KNNWrapper


def validate_knn_parameters(
    parameters: Union[Dict[str, Any], List[Dict[str, Any]]], ml_grid_object: Any
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Validates the `n_neighbors` parameter for KNN classifiers.

    This function ensures that the values for `n_neighbors` do not exceed the
    number of samples in the training data. If a value is too large, it is
    capped at `n_samples - 1`.

    Args:
        parameters (Union[Dict[str, Any], List[Dict[str, Any]]]): The dictionary or list of dictionaries of parameters to validate.
        ml_grid_object (Any): The main pipeline object containing the training
            data (`X_train`).

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: The validated parameters.
    """

    if isinstance(parameters, list):
        for i in range(len(parameters)):
            parameters[i] = validate_knn_parameters(parameters[i], ml_grid_object)
        return parameters

    logger = logging.getLogger("ml_grid")
    # Get the number of samples in the training data
    logger.debug("Validating KNN parameters")
    X_train = ml_grid_object.X_train
    n_samples = X_train.shape[0]
    logger.debug(f"  n_samples: {n_samples}")

    # Get the maximum number of neighbors
    max_neighbors = n_samples - 1
    logger.debug(f"  max_neighbors: {max_neighbors}")

    # Get the n_neighbors values from the parameters
    n_neighbors = parameters.get("n_neighbors")
    logger.debug(f"  Initial n_neighbors: {n_neighbors}")

    # Check if any n_neighbors values are too large
    if n_neighbors is not None and isinstance(n_neighbors, list):
        for i in range(len(n_neighbors)):
            if n_neighbors[i] > max_neighbors:
                logger.debug(
                    f"    Capping n_neighbors[{i}] from {n_neighbors[i]} to {max_neighbors}"
                )
                n_neighbors[i] = max_neighbors

        parameters["n_neighbors"] = n_neighbors
    # Return the validated parameters
    return parameters


def validate_XGB_parameters(
    parameters: Union[Dict[str, Any], List[Dict[str, Any]]], ml_grid_object: Any
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Validates the `max_bin` parameter for XGBoost.

    This function checks that the max_bin values are greater than or equal to 2,
    and if not, it sets them to 2.

    Args:
        parameters (Union[Dict[str, Any], List[Dict[str, Any]]]): The dictionary or list of dictionaries of parameters to validate.
        ml_grid_object (Any): The main pipeline object (currently unused).

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: The validated parameters.
    """

    if isinstance(parameters, list):
        for i in range(len(parameters)):
            parameters[i] = validate_XGB_parameters(parameters[i], ml_grid_object)
        return parameters

    max_bin_array = parameters.get("max_bin")

    if max_bin_array is None or not isinstance(max_bin_array, list):
        return parameters

    # Iterate over each value in the max_bin array
    for i in range(len(max_bin_array)):
        # Check if the value is less than 2
        if max_bin_array[i] < 2:
            # If so, set it to 2
            max_bin_array[i] = 2

    # Update the max_bin array in the parameter combination
    parameters["max_bin"] = max_bin_array

    return parameters


def validate_parameters_helper(
    algorithm_implementation: Any,
    parameters: Union[Dict[str, Any], List[Dict[str, Any]]],
    ml_grid_object: Any,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Dispatches to model-specific validation or performs generic filtering.

    This function first checks for model-specific validation routines (e.g., for
    KNN, XGBoost). If no specific routine is found, it performs a generic
    validation that removes any parameters from the search space that are not
    valid for the given algorithm instance. This prevents `TypeError` exceptions
    from scikit-learn's search classes.

    Args:
        algorithm_implementation (Any): The scikit-learn estimator instance.
        parameters (Union[Dict[str, Any], List[Dict[str, Any]]]): The parameters to validate.
        ml_grid_object (Any): The main pipeline object containing training data.

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: The validated parameters.
    """
    logger = logging.getLogger("ml_grid")

    # --- Model-specific validation ---
    if isinstance(algorithm_implementation, KNeighborsClassifier):
        return validate_knn_parameters(parameters, ml_grid_object)

    try:
        from xgboost import XGBClassifier

        if isinstance(algorithm_implementation, XGBClassifier):
            return validate_XGB_parameters(parameters, ml_grid_object)
    except ImportError:
        logger.debug("XGBoost not installed, skipping XGBoost-specific validation.")
        pass

    # --- Generic fallback: Filter invalid parameters ---
    try:
        valid_params = algorithm_implementation.get_params().keys()
    except Exception:
        logger.warning(
            f"Could not get params for {algorithm_implementation.__class__.__name__}. Skipping generic validation."
        )
        return parameters

    def _filter_dict(param_dict: Dict) -> Dict:
        """Filters a single parameter dictionary."""
        if not isinstance(param_dict, dict):
            return param_dict
        validated_dict = {k: v for k, v in param_dict.items() if k in valid_params}
        removed_keys = set(param_dict.keys()) - set(validated_dict.keys())
        if removed_keys:
            logger.debug(
                f"Removed invalid keys for {algorithm_implementation.__class__.__name__}: {removed_keys}"
            )
        return validated_dict

    if isinstance(parameters, list):
        return [_filter_dict(p) for p in parameters]
    elif isinstance(parameters, dict):
        return _filter_dict(parameters)

    return parameters
