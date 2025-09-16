import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union


def remove_constant_columns(
    X: pd.DataFrame, drop_list: Optional[List[str]] = None, verbose: int = 1
) -> List[str]:
    """Identifies columns in a DataFrame where all values are the same.

    Args:
        X (pd.DataFrame): DataFrame to check for constant columns.
        drop_list (Optional[List[str]], optional): A list of columns already
            marked for dropping. Defaults to None.
        verbose (int, optional): Controls the verbosity of logging. Defaults to 1.

    Returns:
        List[str]: Updated list of columns to drop, including constant columns.

    Raises:
        AssertionError: If X is None.
    """
    try:
        if verbose > 1:
            print("Identifying constant columns")

        assert X is not None, "Null pointer exception: X cannot be None."

        # Initialize drop_list if not provided
        if drop_list is None:
            drop_list = []

        # Identify constant columns
        constant_columns = [col for col in X.columns if X[col].nunique() == 1]

        if constant_columns:
            if verbose > 1:
                print(f"Constant columns identified: {constant_columns}")

            # Add constant columns to drop_list
            drop_list.extend(constant_columns)

    except AssertionError as e:
        print(str(e))
        raise

    except Exception as e:
        print("Unhandled exception:", str(e))
        raise

    return drop_list


def remove_constant_columns_with_debug(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    X_test_orig: Union[pd.DataFrame, np.ndarray],
    verbosity: int = 2,
) -> Tuple[
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
]:
    """Removes constant columns from training and testing datasets.

    This function identifies columns that have zero variance in either the
    training or testing set and removes them from all provided datasets
    (X_train, X_test, X_test_orig). It supports both pandas DataFrames and
    NumPy arrays, including 3D arrays for time series data.

    Args:
        X_train (Union[pd.DataFrame, np.ndarray]): Training feature data.
        X_test (Union[pd.DataFrame, np.ndarray]): Testing feature data.
        X_test_orig (Union[pd.DataFrame, np.ndarray]): Original (unsplit)
            testing feature data.
        verbosity (int, optional): Controls the verbosity of debug messages.
            Defaults to 2.

    Returns:
        Tuple[Union[pd.DataFrame, np.ndarray], ...]: A tuple containing the
        modified X_train, X_test, and X_test_orig datasets with constant
        columns removed.
    """
    if verbosity > 0:
        # Debug message: Initial shapes of X_train, X_test, X_test_orig
        print(f"Initial X_train shape: {X_train.shape}")
        print(f"Initial X_test shape: {X_test.shape}")
        print(f"Initial X_test_orig shape: {X_test_orig.shape}")

    is_pandas = isinstance(X_train, pd.DataFrame)

    if is_pandas:
        # Original logic for pandas DataFrames
        train_variances = X_train.var(axis=0)
        if verbosity > 1:
            print(f"Variance of X_train columns:\n{train_variances}")

        constant_columns_train = train_variances[train_variances == 0].index
        if verbosity > 0:
            print(f"Constant columns in X_train: {list(constant_columns_train)}")

        test_variances = X_test.var(axis=0)
        if verbosity > 1:
            print(f"Variance of X_test columns:\n{test_variances}")

        constant_columns_test = test_variances[test_variances == 0].index
        if verbosity > 0:
            print(f"Constant columns in X_test: {list(constant_columns_test)}")

        constant_columns = constant_columns_train.union(constant_columns_test)

        X_train = X_train.loc[:, ~X_train.columns.isin(constant_columns)]
        X_test = X_test.loc[:, ~X_test.columns.isin(constant_columns)]
        X_test_orig = X_test_orig.loc[:, ~X_test_orig.columns.isin(constant_columns)]
    else:  # Handle numpy arrays
        # Determine variance calculation axis based on dimensions
        if X_train.ndim == 3:
            # For 3D time series data (e.g., from aeon: samples, features, timesteps),
            # calculate variance for each feature across samples and timesteps.
            var_axis = (0, 2)
        else:
            # For 2D data, calculate variance across samples (axis 0).
            var_axis = 0

        train_variances = X_train.var(axis=var_axis)
        constant_indices_train = np.where(train_variances == 0)[0]
        if verbosity > 0:
            print(f"Constant feature indices in X_train: {list(constant_indices_train)}")

        test_variances = X_test.var(axis=var_axis)
        constant_indices_test = np.where(test_variances == 0)[0]
        if verbosity > 0:
            print(f"Constant feature indices in X_test: {list(constant_indices_test)}")

        # Combine indices of constant features from both train and test sets
        constant_indices = np.union1d(constant_indices_train, constant_indices_test)

        # Create a boolean mask for features to keep
        num_features = X_train.shape[1]
        keep_mask = np.ones(num_features, dtype=bool)
        keep_mask[constant_indices] = False

        # Apply the mask to remove constant features
        if X_train.ndim == 3:
            X_train = X_train[:, keep_mask, :]
            X_test = X_test[:, keep_mask, :]
            X_test_orig = X_test_orig[:, keep_mask, :]
        else:  # 2D array
            X_train = X_train[:, keep_mask]
            X_test = X_test[:, keep_mask]
            X_test_orig = X_test_orig[:, keep_mask]

    if verbosity > 0:
        # Debug message: Shape after removing constant columns from X_train, X_test, X_test_orig
        print(f"Shape of X_train after removing constant columns: {X_train.shape}")
        print(f"Shape of X_test after removing constant columns: {X_test.shape}")
        print(f"Shape of X_test_orig after removing constant columns: {X_test_orig.shape}")

    # Return the modified X_train, X_test, and X_test_orig, with y_test_orig unchanged
    return X_train, X_test, X_test_orig

# Example usage with verbosity level 2 (most verbose)
# X_train, X_test, X_test_orig = remove_constant_columns_with_debug(X_train, X_test, X_test_orig, verbosity=2)
