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