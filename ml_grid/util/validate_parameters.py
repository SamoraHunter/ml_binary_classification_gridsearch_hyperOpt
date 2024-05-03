from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ml_grid.model_classes.knn_gpu_classifier_class import knn__gpu_wrapper_class
from ml_grid.model_classes.knn_wrapper_class import KNNWrapper
from ml_grid.util.global_params import global_parameters


def validate_knn_parameters(parameters, ml_grid_object):
    """Validate the parameters for KNN.

    This function makes sure that the values for n_neighbors are within
    the range of the number of samples in the training data. If a value
    is greater than the maximum number of samples, it is reduced to be
    equal to the maximum number of samples minus one.

    Parameters
    ----------
    parameters : dict
        The parameters to be validated.
    ml_grid_object : object
        The object from which the training data is obtained.

    Returns
    -------
    dict
        The validated parameters.
    """

    # Get the number of samples in the training data
    print("Validating KNN parameters")
    X_train = ml_grid_object.X_train
    n_samples = X_train.shape[0]
    print("  n_samples: ", n_samples)

    # Get the maximum number of neighbors
    max_neighbors = n_samples - 1
    print("  max_neighbors: ", max_neighbors)

    # Get the n_neighbors values from the parameters
    n_neighbors = parameters.get("n_neighbors")
    print("  n_neighbors: ", n_neighbors)

    # Check if any n_neighbors values are too large
    if n_neighbors is not None:
        for i in range(len(n_neighbors)):
            if n_neighbors[i] > max_neighbors:
                print("  n_neighbors[{}] is greater than max_neighbors".format(i))

                # If so, reduce the value to be within the allowed range
                print(
                    "    Reducing n_neighbors[{}] from {} to {}".format(
                        i, n_neighbors[i], max_neighbors
                    )
                )
                n_neighbors[i] = max_neighbors

    parameters['n_neighbors'] = n_neighbors
    # Return the validated parameters
    return parameters



def validate_XGB_parameters(parameters, ml_grid_object):
    """
    Validate the parameters for XGBoost.

    This function checks that the max_bin values are greater than or equal to 2,
    and if not, it sets them to 2.

    Parameters
    ----------
    parameters : dict
        The parameters to be validated.
    ml_grid_object : object
        The object from which the training data is obtained.

    Returns
    -------
    dict
        The validated parameters.
    """

    max_bin_array = parameters.get("max_bin")

    # Iterate over each value in the max_bin array
    for i in range(len(max_bin_array)):
        # Check if the value is less than 2
        if max_bin_array[i] < 2:
            # If so, set it to 2
            max_bin_array[i] = 2

    # Update the max_bin array in the parameter combination
    parameters['max_bin'] = max_bin_array

    return parameters


def validate_parameters_helper(algorithm_implementation, parameters, ml_grid_object):

    if type(algorithm_implementation) == KNeighborsClassifier:

        parameters = validate_knn_parameters(parameters, ml_grid_object)

        return parameters
    
    elif type(algorithm_implementation) == KNNWrapper:

        parameters = validate_knn_parameters(parameters, ml_grid_object)

        return parameters
    
    elif type(algorithm_implementation) == knn__gpu_wrapper_class:

        parameters = validate_knn_parameters(parameters, ml_grid_object)

        return parameters

    
    elif type(algorithm_implementation) == XGBClassifier:
        parameters = validate_XGB_parameters(parameters, ml_grid_object)

        return parameters

    else:
        return parameters
    


