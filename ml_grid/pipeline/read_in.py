import random
import pandas as pd
import numpy as np

import polars as pl

class read:
    def __init__(self, input_filename, use_polars=False):
        filename = input_filename
        print(f"Init main >read on {filename}")
        if use_polars:
            try:
                self.raw_input_data = pl.read_csv(filename, ignore_errors=True)
                self.raw_input_data = self.raw_input_data.to_pandas()
            except Exception as e:
                print(f"Error reading with Polars: {e}")
                print(f"Error reading with Polars: {e}")
                print("Trying to read with Pandas...")
                try:
                    self.raw_input_data = pd.read_csv(filename)
                except Exception as e:
                    print(f"Error reading with Pandas: {e}")
                    self.raw_input_data = pd.DataFrame()
        else:
            try:
                self.raw_input_data = pd.read_csv(filename)
            except Exception as e:
                print(f"Error reading with Pandas: {e}")
                self.raw_input_data = pd.DataFrame()



class read_sample:
    def __init__(
        self, input_filename: str, test_sample_n: int, column_sample_n: int
    ) -> None:
        """
        Initialize the class with the input filename, test sample number, and column sample number.
        
        This class is designed to read in a sample from the input data,
        where the number of rows and columns to read in can be specified.
        
        The rows are read in randomly,
        and the columns are read in randomly IF the number of columns to read in is less than the total number of columns in the file.
        
        The class will raise an error if the outcome variable does not have both classes after sampling.
        
        :param input_filename: str, the filename of the input data
        :param test_sample_n: int, the number of rows to read from the input data
        :param column_sample_n: int, the number of columns to read from the input data
        :return: None
        """
        self.filename = input_filename

        # The columns that are necessary to be in the input data
        necessary_columns = ["outcome_var_1", "age", "male"]

        # Get the total number of rows in the CSV file
        total_rows = sum(1 for line in open(self.filename))

        # Calculate the number of rows to skip to achieve random sampling on read in
        skip_rows = np.random.choice(np.arange(1, total_rows), total_rows - test_sample_n, replace=False)

        # Read column names from the file
        all_columns = pd.read_csv(self.filename, nrows=1).columns.tolist()

        # Select the necessary columns from the file
        necessary_columns = [col for col in necessary_columns if col in all_columns]

        # Select the remaining columns from the file
        remaining_columns = [col for col in all_columns if col not in necessary_columns]

        # Calculate the maximum number of additional columns that can be selected
        max_additional_columns = test_sample_n - len(necessary_columns)

        # Sample the remaining columns
        # If the number of columns to read in is less than the total number of columns in the file
        selected_additional_columns = random.sample(remaining_columns, min(len(remaining_columns), max_additional_columns))

        # Combine the necessary and selected additional columns
        selected_columns = necessary_columns + selected_additional_columns

        print(f"Init main > read_sample on {self.filename}")

        # If both test_sample_n and column_sample_n are 0
        # Read in all columns and all rows
        if test_sample_n == 0 and column_sample_n == 0:
            self.raw_input_data = pd.read_csv(self.filename) # Read in all columns and all rows

        # If test_sample_n is 0 but column_sample_n is greater than 0
        # Read in all rows but sample the columns
        elif test_sample_n == 0 and column_sample_n > 0:
            # Read in the file with the selected columns
            self.raw_input_data = pd.read_csv(self.filename, usecols=selected_columns)

        # If column_sample_n is 0 but test_sample_n is greater than 0
        # Read in a sample of the rows but all columns
        elif column_sample_n == 0 and test_sample_n > 0:
            # Read in the file with the selected rows
            self.raw_input_data = pd.read_csv(
                self.filename,
                skiprows=skip_rows,
            ) 

        # If both test_sample_n and column_sample_n are greater than 0
        # Read in a sample of the rows and columns
        else:
            # Read in the file with the selected rows and columns
            self.raw_input_data = pd.read_csv(
                self.filename,
                skiprows=skip_rows,
                usecols=selected_columns,
            )  

        # Check if the outcome variable has both classes
        if self.raw_input_data is not None and 'outcome_var_1' in self.raw_input_data.columns:
            classes = self.raw_input_data['outcome_var_1'].unique()
            if len(classes) < 2:
                raise ValueError("Outcome variable does not have both classes post sampling.")
