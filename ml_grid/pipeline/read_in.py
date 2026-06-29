import logging
import random

import pandas as pd
import polars as pl

# Module-level random generator for isolation from test state
_random_generator = random.Random()


class read:
    """Reads a CSV file into a pandas DataFrame, with an option to use Polars for faster reading."""

    def __init__(self, input_filename: str, use_polars: bool = False):
        """Initializes the read class and loads the data.

        Args:
            input_filename (str): The path to the input CSV file.
            use_polars (bool, optional): If True, attempts to read the CSV using
                the Polars library and converts it to a pandas DataFrame.
                Falls back to pandas if Polars fails. Defaults to False.
        """
        logger = logging.getLogger("ml_grid")
        filename = input_filename
        logger.info(f"Reading data from {filename}")
        if use_polars:
            try:
                self.raw_input_data = pl.read_csv(filename, ignore_errors=True)
                self.raw_input_data = self.raw_input_data.to_pandas()
            except Exception as e:
                logger.warning(f"Error reading with Polars: {e}")
                logger.info("Falling back to pandas...")
                try:
                    self.raw_input_data = pd.read_csv(filename)
                except Exception as e:
                    logger.error(f"Error reading with Pandas: {e}")
                    self.raw_input_data = pd.DataFrame()
        else:
            try:
                self.raw_input_data = pd.read_csv(filename)
            except Exception as e:
                logger.error(f"Error reading with Pandas: {e}")
                self.raw_input_data = pd.DataFrame()


class read_sample:
    def __init__(
        self, input_filename: str, test_sample_n: int, column_sample_n: int
    ) -> None:
        """Initializes the read_sample class and loads a data sample.

        This class reads a random sample of rows and/or columns from a CSV file.
        It ensures that certain `necessary_columns` are always included if they
        exist in the source file.

        Note:
            The column sampling logic (`max_additional_columns`) appears to be
            based on the number of rows to sample (`test_sample_n`) rather than
            the number of columns (`column_sample_n`), which may be unintended.
            The functionality has been preserved as is.

        Args:
            input_filename (str): The path to the input CSV file.
            test_sample_n (int): The number of rows to randomly sample. If 0,
                all rows are read.
            column_sample_n (int): The number of columns to randomly sample,
                in addition to the `necessary_columns`.

        Raises:
            ValueError: If the 'outcome_var_1' column does not contain at least
                two unique classes after sampling.
        """
        logger = logging.getLogger("ml_grid")
        self.filename = input_filename

        # The columns that are necessary to be in the input data
        necessary_columns = ["outcome_var_1", "age", "male"]

        # Read column names from the file
        all_columns = pd.read_csv(self.filename, nrows=1).columns.tolist()

        # Select the necessary columns from the file
        necessary_columns = [col for col in necessary_columns if col in all_columns]

        # Select the remaining columns from the file
        remaining_columns = [col for col in all_columns if col not in necessary_columns]

        logger.info(f"Reading data sample from {self.filename}")

        # If both test_sample_n and column_sample_n are 0
        # Read in all columns and all rows
        if test_sample_n == 0 and column_sample_n == 0:
            self.raw_input_data = pd.read_csv(
                self.filename
            )  # Read in all columns and all rows

        # If test_sample_n is 0 but column_sample_n is greater than 0
        # Read in all rows but sample the columns
        elif test_sample_n == 0 and column_sample_n > 0:
            # Calculate the maximum number of additional columns that can be selected
            max_additional_columns = column_sample_n

            # Sample the remaining columns
            if len(remaining_columns) > 0 and max_additional_columns > 0:
                selected_additional_columns = _random_generator.sample(
                    remaining_columns,
                    min(len(remaining_columns), max_additional_columns),
                )
            else:
                selected_additional_columns = []

            # Combine the necessary and selected additional columns
            selected_columns = necessary_columns + selected_additional_columns

            # Read in the file with the selected columns
            self.raw_input_data = pd.read_csv(self.filename, usecols=selected_columns)

        # If column_sample_n is 0 but test_sample_n is greater than 0
        # Read in a sample of the rows but all columns
        elif column_sample_n == 0 and test_sample_n > 0:
            # Get the total number of rows in the CSV file (including header)
            total_rows = sum(1 for line in open(self.filename))

            # Calculate data rows available (excluding header) and skip count
            num_data_rows = total_rows - 1
            skip_count = max(0, num_data_rows - test_sample_n)

            if skip_count > 0:
                skip_rows = _random_generator.sample(range(1, total_rows), skip_count)
                self.raw_input_data = pd.read_csv(
                    self.filename,
                    skiprows=skip_rows,
                )
            else:
                # If we want all or more rows than exist, read all
                self.raw_input_data = pd.read_csv(self.filename)

        # If both test_sample_n and column_sample_n are greater than 0
        # Read in a sample of the rows and columns
        else:
            # Get the total number of rows in the CSV file (including header)
            total_rows = sum(1 for line in open(self.filename))

            # Calculate data rows available (excluding header) and skip count
            num_data_rows = total_rows - 1
            skip_count = max(0, num_data_rows - test_sample_n)

            if skip_count > 0:
                skip_rows = _random_generator.sample(range(1, total_rows), skip_count)
            else:
                skip_rows = []

            # Calculate the maximum number of additional columns that can be selected
            max_additional_columns = min(column_sample_n, len(remaining_columns))

            # Sample the remaining columns
            if len(remaining_columns) > 0 and max_additional_columns > 0:
                selected_additional_columns = _random_generator.sample(
                    remaining_columns,
                    min(len(remaining_columns), max_additional_columns),
                )
            else:
                selected_additional_columns = []

            # Combine the necessary and selected additional columns
            selected_columns = necessary_columns + selected_additional_columns

            # Read in the file with the selected rows and columns
            if len(skip_rows) > 0:
                self.raw_input_data = pd.read_csv(
                    self.filename,
                    skiprows=skip_rows,
                    usecols=selected_columns,
                )
            else:
                self.raw_input_data = pd.read_csv(
                    self.filename,
                    usecols=selected_columns,
                )

        # Check if the outcome variable has both classes
        if (
            self.raw_input_data is not None
            and "outcome_var_1" in self.raw_input_data.columns
        ):
            classes = self.raw_input_data["outcome_var_1"].unique()
            if len(classes) < 2:
                raise ValueError(
                    "Outcome variable does not have both classes post sampling."
                )
