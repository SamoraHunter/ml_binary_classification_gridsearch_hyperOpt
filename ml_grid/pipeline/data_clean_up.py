import re

import pandas as pd

from ml_grid.util.global_params import global_parameters


class clean_up_class:

    def __init__(self):

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        self.rename_cols = self.global_params.rename_cols

        # print mass debug statement for cleaning procedures?

        # pass

    def handle_duplicated_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops duplicated columns from X, returns a copy.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to drop duplicated columns from.

        Returns
        -------
        pd.DataFrame
            Copy of X with duplicated columns dropped.

        """

        try:
            if self.verbose > 1:
                print("dropping duplicated columns")

            assert X is not None, "Null pointer exception: X cannot be None."

            X = X.loc[:, ~X.columns.duplicated()].copy()

            assert (
                X is not None
            ), "Null pointer exception: X cannot be None after dropping duplicated columns."

        except AssertionError as e:
            print(str(e))
            raise

        except Exception as e:
            print("Unhandled exception:", str(e))
            raise

        return X

    def screen_non_float_types(self, X):

        if self.verbose > 1:
            print("Screening for non float data types:")
            # types = []
            for col in X.columns:
                if X[col].dtype != int and X[col].dtype != float:
                    print(col)

    def handle_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to remove bad characters, xgb related.

        This function is used to rename columns in a DataFrame (X) that have
        characters in their names that are not supported by XGBoost.

        The renaming is done based on a regular expression that matches "[", "]",
        "<" in the column name. The renaming replaces these characters with "_".

        The renaming is only done if the rename_cols parameter is set to True.
        Otherwise, the columns are not renamed.

        If the column name contains any of the following characters "[", "]", "<"
        then it is renamed. Otherwise, the column name is not changed.

        The reason for this is that XGBoost will throw an error if it
        encounters any of these characters in the column name.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with columns to be renamed.

        Returns
        -------
        pd.DataFrame
            Copy of X with renamed columns.

        """

        if self.rename_cols:

            # define a regular expression that matches "[", "]", "<" in the
            # column name
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)

            # create a new list of column names
            new_col_names = []

            # loop through all the column names in X
            for col in X.columns.values:

                # check if the column name contains any of the characters "[", "]",
                # "<" using the any() function
                if any(X in str(col) for X in set(("[", "]", "<"))):

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
