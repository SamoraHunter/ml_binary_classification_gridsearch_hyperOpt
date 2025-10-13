import re
from typing import List
import logging

import pandas as pd

from ml_grid.util.global_params import global_parameters


class clean_up_class:
    """A class for cleaning and preparing DataFrame columns."""

    def __init__(self):
        """Initializes the clean_up_class."""
        self.global_params = global_parameters

        self.logger = logging.getLogger('ml_grid')

        self.verbose = self.global_params.verbose

        self.rename_cols = self.global_params.rename_cols

    def handle_duplicated_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicated columns from a DataFrame.

        Args:
            X (pd.DataFrame): DataFrame to drop duplicated columns from.

        Returns:
            pd.DataFrame: A copy of X with duplicated columns dropped.

        Raises:
            AssertionError: If X is None before or after processing.
        """
        try:
            if self.verbose > 1:
                self.logger.info("Dropping duplicated columns")

            assert X is not None, "Null pointer exception: X cannot be None."

            X = X.loc[:, ~X.columns.duplicated()].copy()

            assert X is not None, (
                "Null pointer exception: X cannot be None after dropping "
                "duplicated columns."
            )

        except AssertionError as e:
            self.logger.error(str(e))
            raise

        except Exception as e:
            self.logger.error(f"Unhandled exception: {e}", exc_info=True)
            raise

        return X

    def screen_non_float_types(self, X: pd.DataFrame) -> None:
        """Screens and prints columns that are not of float or int type.

        Args:
            X (pd.DataFrame): The DataFrame to screen.
        """
        if self.verbose > 1:
            self.logger.info("Screening for non float data types:")
            for col in X.columns:
                if X[col].dtype != int and X[col].dtype != float:
                    self.logger.info(col)

    def handle_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """Renames columns to remove characters unsupported by some ML models.

        This function renames columns in a DataFrame (X) that contain
        characters like '[', ']', or '<', which can cause issues with models
        like XGBoost. These characters are replaced with underscores.

        The renaming is controlled by the `self.rename_cols` attribute.

        Args:
            X (pd.DataFrame): DataFrame with columns to be potentially renamed.

        Returns:
            pd.DataFrame: A copy of X with renamed columns if applicable.
        """
        if self.rename_cols:
            # define a regular expression that matches "[", "]", "<" in the
            # column name
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)

            # create a new list of column names
            new_col_names: List[str] = []

            # loop through all the column names in X
            for col in X.columns.values:
                # check if the column name contains any of the characters "[", "]",
                # "<" using the any() function
                if any(char in str(col) for char in {"[", "]", "<"}):
                    # if it does, rename the column by replacing the characters
                    # "[", "]", "<" with "_"
                    new_col_names.append(regex.sub("_", col))
                # if the column name does not contain any of the characters "[", "]",
                # "<", keep the original column name
                else:
                    new_col_names.append(col)

            # set the column names of X to be the new list of names created above
            X.columns = new_col_names

        # return a copy of X with the new column names
        return X
