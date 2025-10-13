import random
from typing import Any, Dict, Tuple, Union
import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def get_data_split(
    X: pd.DataFrame, y: pd.Series, local_param_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Splits data into train and test sets, with optional resampling.

    This function splits the input data (X, y) into training and testing sets.
    It can perform no resampling, undersampling, or oversampling based on the
    'resample' key in `local_param_dict`. The data is first split into a
    preliminary train/test set, and then the preliminary training set is
    further split to create the final train/test sets for model evaluation,
    while the original test set is preserved for final validation.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target variable.
        local_param_dict (Dict[str, Any]): A dictionary of parameters,
            including the 'resample' strategy ('undersample', 'oversample',
            or None).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
        A tuple containing:
            - X_train: Features for training.
            - X_test: Features for testing.
            - y_train: Target variable for training.
            - y_test: Target variable for testing.
            - X_test_orig: Original features for validation.
            - y_test_orig: Original target variable for validation.
    """
    logger = logging.getLogger('ml_grid')
    random.seed(1234)
    np.random.seed(1234)

    # Check if data is valid
    if not is_valid_shape(X):
        local_param_dict["resample"] = None
        logger.warning("Input data is not 2D, overriding resample strategy to None.")

    # No resampling
    # First, split into a preliminary training set and a final hold-out test set.
    # This original test set will NOT be resampled.
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )

    # Now, handle resampling ONLY on the preliminary training set (X_train_orig)
    if local_param_dict.get("resample") == "undersample":
        # Store original column names and y name to reconstruct DataFrame after resampling
        original_columns = X.columns
        y_name = y.name

        # Undersample data
        rus = RandomUnderSampler(random_state=1)
        X_train_res, y_train_res = rus.fit_resample(X_train_orig, y_train_orig)
        
        # Reconstruct DataFrame and Series to ensure correct indices and names
        X_train_processed = pd.DataFrame(X_train_res, columns=original_columns)
        y_train_processed = pd.Series(y_train_res, name=y_name)
        

    # Oversampling
    elif local_param_dict.get("resample") == "oversample":
        # Store original column names to reconstruct DataFrame after resampling
        original_columns = X_train_orig.columns
        y_name = y_train_orig.name

        # Oversample training set
        sampling_strategy = 1
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=1)
        X_train_res, y_train_res = ros.fit_resample(X_train_orig, y_train_orig)
        
        # Reconstruct DataFrame and Series
        X_train_processed = pd.DataFrame(X_train_res, columns=original_columns)
        y_train_processed = pd.Series(y_train_res, name=y_name)
    
    else: # No resampling
        X_train_processed = X_train_orig
        y_train_processed = y_train_orig

    # Finally, split the (potentially resampled) training data into the final
    # training and validation sets for the grid search.
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_processed,
        y_train_processed,
        test_size=0.25,
        random_state=1,
        stratify=y_train_processed,
    )

    return X_train, X_test, y_train, y_test, X_test_orig, y_test_orig


def is_valid_shape(input_data: Union[np.ndarray, pd.DataFrame]) -> bool:
    """Checks if the input data is a 2-dimensional array or DataFrame.

    This is used to validate data before resampling, as some resampling
    techniques may not work with other data shapes.

    Args:
        input_data (Union[np.ndarray, pd.DataFrame]): The data to check.

    Returns:
        bool: True if the data is 2-dimensional, False otherwise.
    """
    # Check if input_data is a numpy array
    if isinstance(input_data, np.ndarray):
        # If it's a numpy array, directly check its number of dimensions
        return input_data.ndim == 2

    # Check if input_data is a pandas DataFrame
    elif isinstance(input_data, pd.DataFrame):
        # If it's a DataFrame, convert it to a numpy array and then check its shape
        input_array = input_data.values
        return input_array.ndim == 2

    else:
        # Input data is neither a numpy array nor a pandas DataFrame
        return False