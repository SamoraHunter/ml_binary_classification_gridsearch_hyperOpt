"""Functions to validate model-specific hyperparameters before grid search."""

import logging
from typing import Any, Dict

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# from ml_grid.model_classes.knn_gpu_classifier_class import KNNGpuWrapperClass
# from ml_grid.model_classes.knn_wrapper_class import KNNWrapper


def validate_knn_parameters(
    parameters: Dict[str, Any], ml_grid_object: Any
) -> Dict[str, Any]:
    """Validates the `n_neighbors` parameter for KNN classifiers.

    This function ensures that the values for `n_neighbors` do not exceed the
    number of samples in the training data. If a value is too large, it is
    capped at `n_samples - 1`.

    Args:
        parameters (Dict[str, Any]): The dictionary of parameters to validate.
        ml_grid_object (Any): The main pipeline object containing the training
            data (`X_train`).

    Returns:
        Dict[str, Any]: The validated parameters dictionary.
    """

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
    if n_neighbors is not None:
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
    parameters: Dict[str, Any], ml_grid_object: Any
) -> Dict[str, Any]:
    """Validates the `max_bin` parameter for XGBoost.

    This function checks that the max_bin values are greater than or equal to 2,
    and if not, it sets them to 2.

    Args:
        parameters (Dict[str, Any]): The dictionary of parameters to validate.
        ml_grid_object (Any): The main pipeline object (currently unused).

    Returns:
        Dict[str, Any]: The validated parameters dictionary.
    """

    max_bin_array = parameters.get("max_bin")

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
    algorithm_implementation: Any, parameters: Dict[str, Any], ml_grid_object: Any
) -> Dict[str, Any]:
    """Dispatches to the correct parameter validation function based on algorithm type.

    Args:
        algorithm_implementation (Any): The scikit-learn estimator instance.
        parameters (Dict[str, Any]): The dictionary of parameters to validate.
        ml_grid_object (Any): The main pipeline object containing training data.

    Returns:
        Dict[str, Any]: The validated parameters dictionary.
    """

    if type(algorithm_implementation) == KNeighborsClassifier:

        parameters = validate_knn_parameters(parameters, ml_grid_object)

        return parameters

    # elif type(algorithm_implementation) == KNNWrapper:

    #     parameters = validate_knn_parameters(parameters, ml_grid_object)

    #     return parameters

    # elif isinstance(algorithm_implementation, KNNGpuWrapperClass):

    #     parameters = validate_knn_parameters(parameters, ml_grid_object)

    #     return parameters

    elif isinstance(algorithm_implementation, XGBClassifier):
        parameters = validate_XGB_parameters(parameters, ml_grid_object)

        return parameters

    else:
        return parameters
