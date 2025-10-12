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

    This function identifies columns that have zero variance in the
    training set and removes them from all provided datasets
    (X_train, X_test, X_test_orig). It supports both pandas DataFrames and
    NumPy arrays, including 3D arrays for time series data.
    
    IMPORTANT: Only checks X_train for constant columns to prevent data leakage.
    A column is constant if it has <= 1 unique value in X_train.

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
        # Identify constant columns in X_train
        # A column is constant if it has 1 or fewer unique values (excluding NaN)
        constant_columns = []
        
        for col in X_train.columns:
            try:
                # Use nunique() without dropna to be more conservative
                # A column with all NaN is also constant and should be dropped
                n_unique = X_train[col].nunique(dropna=False)
                
                if n_unique <= 1:
                    constant_columns.append(col)
                    if verbosity > 1:
                        print(f"Column '{col}' is constant: nunique={n_unique}, sample values: {X_train[col].head()}")
                # Additional check for numeric columns with zero variance
                elif pd.api.types.is_numeric_dtype(X_train[col]):
                    try:
                        col_std = X_train[col].std()
                        if col_std == 0 or (pd.notna(col_std) and np.isclose(col_std, 0)):
                            constant_columns.append(col)
                            if verbosity > 1:
                                print(f"Column '{col}' has zero variance: std={col_std}")
                    except Exception as e:
                        if verbosity > 1:
                            print(f"Could not calculate std for column '{col}': {e}")
            except Exception as e:
                if verbosity > 1:
                    print(f"Error checking column '{col}': {e}")
        
        if verbosity > 1:
            print(f"\nUnique value counts in X_train:")
            for col in X_train.columns:
                print(f"  {col}: {X_train[col].nunique(dropna=False)} unique values")
        
        if verbosity > 0:
            print(f"\nConstant columns identified in X_train: {constant_columns}")
            print(f"Number of constant columns to remove: {len(constant_columns)}")
        
        if constant_columns:
            X_train = X_train.drop(columns=constant_columns, errors='ignore')
            X_test = X_test.drop(columns=constant_columns, errors='ignore')
            X_test_orig = X_test_orig.drop(columns=constant_columns, errors='ignore')
            
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

        # A feature is constant if it has no variance in the training set.
        # We should not consider the test set variance, as a small test set
        # might misleadingly have constant features.
        constant_indices = constant_indices_train
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