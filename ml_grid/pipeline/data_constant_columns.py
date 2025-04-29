import pandas as pd
from typing import List, Optional

def remove_constant_columns(X: pd.DataFrame, drop_list: Optional[List[str]] = None, verbose: int = 1) -> List[str]:
    """
    Identifies columns in X where all values are the same (constant) and returns their names.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to check for constant columns.
    drop_list : List[str], optional
        List of columns already marked for dropping. Default is None.
    verbose : int, optional
        Controls the verbosity of logging. Default is 1.

    Returns
    -------
    List[str]
        Updated list of columns to drop, including constant columns.
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

def remove_constant_columns_with_debug(X_train, X_test, X_test_orig, verbosity=2):
    if verbosity > 0:
        # Debug message: Initial shapes of X_train, X_test, X_test_orig
        print(f"Initial X_train shape: {X_train.shape}")
        print(f"Initial X_test shape: {X_test.shape}")
        print(f"Initial X_test_orig shape: {X_test_orig.shape}")
    
    # Calculate the variance for each column in X_train
    train_variances = X_train.var(axis=0)
    if verbosity > 1:
        print(f"Variance of X_train columns:\n{train_variances}")
    
    # Identify and remove constant columns in X_train
    constant_columns_train = train_variances[train_variances == 0].index
    if verbosity > 0:
        print(f"Constant columns in X_train: {list(constant_columns_train)}")
    
    # Calculate the variance for each column in X_test
    test_variances = X_test.var(axis=0)
    if verbosity > 1:
        print(f"Variance of X_test columns:\n{test_variances}")
    
    # Identify constant columns in X_test
    constant_columns_test = test_variances[test_variances == 0].index
    if verbosity > 0:
        print(f"Constant columns in X_test: {list(constant_columns_test)}")
    
    # Combine constant columns from both X_train and X_test
    constant_columns = constant_columns_train.union(constant_columns_test)
    
    # Remove the constant columns from both X_train and X_test
    X_train = X_train.loc[:, ~X_train.columns.isin(constant_columns)]
    X_test = X_test.loc[:, ~X_test.columns.isin(constant_columns)]
    
    # Also remove the same constant columns from X_test_orig
    X_test_orig = X_test_orig.loc[:, ~X_test_orig.columns.isin(constant_columns)]
    
    if verbosity > 0:
        # Debug message: Shape after removing constant columns from X_train, X_test, X_test_orig
        print(f"Shape of X_train after removing constant columns: {X_train.shape}")
        print(f"Shape of X_test after removing constant columns: {X_test.shape}")
        print(f"Shape of X_test_orig after removing constant columns: {X_test_orig.shape}")
    
    # Return the modified X_train, X_test, and X_test_orig, with y_test_orig unchanged
    return X_train, X_test, X_test_orig

# Example usage with verbosity level 2 (most verbose)
# X_train, X_test, X_test_orig = remove_constant_columns_with_debug(X_train, X_test, X_test_orig, verbosity=2)
