import logging
import random
from typing import Any, Dict, Tuple, Union

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
    logger = logging.getLogger("ml_grid")
    random.seed(1234)
    np.random.seed(1234)

    # Check if data is valid
    if not is_valid_shape(X):
        local_param_dict["resample"] = None
        logger.warning("Input data is not 2D, overriding resample strategy to None.")

    # --- SAFEGUARD for Stratification ---
    # Check if any class has fewer than 2 samples for stratified splitting
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    use_stratify = min_class_count >= 2

    if not use_stratify:
        logger.warning(
            f"Cannot use stratified split: smallest class has only {min_class_count} sample(s). "
            f"Class distribution: {class_counts.to_dict()}. Using random split instead."
        )
        # Also disable resampling since we can't properly balance with so few samples
        if local_param_dict.get("resample") is not None:
            logger.warning(
                "Disabling resampling due to insufficient samples in minority class."
            )
            local_param_dict["resample"] = None

    # First, split into a preliminary training set and a final hold-out test set.
    # This original test set will NOT be resampled.
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y if use_stratify else None
    )

    # --- SAFEGUARD for Resampling ---
    # Check if the minority class in the training set is too small for resampling.
    # imblearn samplers require at least 2 samples in the minority class.
    minority_class_count = y_train_orig.value_counts().min()
    if minority_class_count < 2 and local_param_dict.get("resample") is not None:
        logger.warning(
            f"Minority class has only {minority_class_count} sample(s) in the training fold. "
            f"Disabling resampling to prevent errors."
        )
        local_param_dict["resample"] = None

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
        ros = RandomOverSampler(sampling_strategy="auto", random_state=1)
        X_train_res, y_train_res = ros.fit_resample(X_train_orig, y_train_orig)

        # Reconstruct DataFrame and Series
        X_train_processed = pd.DataFrame(X_train_res, columns=original_columns)
        y_train_processed = pd.Series(y_train_res, name=y_name)

    else:  # No resampling
        X_train_processed = X_train_orig
        y_train_processed = y_train_orig

    # Check again if we can stratify the second split
    train_class_counts = y_train_processed.value_counts()
    min_train_class_count = train_class_counts.min()
    use_stratify_second = min_train_class_count >= 2

    if not use_stratify_second:
        logger.warning(
            f"Cannot use stratified split for train/validation: smallest class has only "
            f"{min_train_class_count} sample(s). Using random split instead."
        )

    # Finally, split the (potentially resampled) training data into the final
    # training and validation sets for the grid search.
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_processed,
        y_train_processed,
        test_size=0.25,
        random_state=1,
        stratify=y_train_processed if use_stratify_second else None,
    )

    # --- Fallback for single-class training set ---
    # If the random split resulted in a training set with only 1 class (but we had 2+ available),
    # we attempt to move a sample from the test set to the training set to prevent model failure.
    if y_train.nunique() < 2 and y_train_processed.nunique() >= 2:
        logger.warning(
            "y_train contains only 1 class after split. Attempting to move a sample from X_test to X_train to ensure class presence."
        )
        missing_classes = set(y_train_processed.unique()) - set(y_train.unique())
        for missing_cls in missing_classes:
            # Find candidates in test set
            candidates = y_test[y_test == missing_cls]
            if not candidates.empty:
                idx_to_move = candidates.index[0]

                # Move from test to train
                X_train = pd.concat([X_train, X_test.loc[[idx_to_move]]])
                y_train = pd.concat([y_train, y_test.loc[[idx_to_move]]])

                X_test = X_test.drop(idx_to_move)
                y_test = y_test.drop(idx_to_move)

                logger.info(
                    f"Moved sample {idx_to_move} (class {missing_cls}) from test to train."
                )
                break  # Only need one sample to satisfy "at least 2 classes"

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
