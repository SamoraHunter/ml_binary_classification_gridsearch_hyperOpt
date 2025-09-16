import re
import random
from typing import Any, Dict, List, Optional
import warnings

import pandas as pd
from IPython.display import display
from sklearn.exceptions import ConvergenceWarning

from ml_grid.pipeline import read_in
from ml_grid.pipeline.column_names import get_pertubation_columns
from ml_grid.pipeline.data_clean_up import clean_up_class
from ml_grid.pipeline.data_constant_columns import remove_constant_columns, remove_constant_columns_with_debug
from ml_grid.pipeline.data_correlation_matrix import handle_correlation_matrix
from ml_grid.pipeline.data_feature_importance_methods import feature_importance_methods
from ml_grid.pipeline.data_outcome_list import handle_outcome_list
from ml_grid.pipeline.data_percent_missing import handle_percent_missing
from ml_grid.pipeline.data_plot_split import plot_pie_chart_with_counts
from ml_grid.pipeline.data_scale import data_scale_methods
from ml_grid.pipeline.data_train_test_split import get_data_split, is_valid_shape
from ml_grid.pipeline.logs_project_folder import log_folder
from ml_grid.util.global_params import global_parameters

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class pipe:
    """Represents a single data processing pipeline permutation.

    This class reads data, applies a series of cleaning and feature selection
    steps based on a dictionary of parameters, and prepares the data for model
    training and evaluation. The resulting object holds the processed data and
    can be passed to classifier methods.
    """

    def __init__(
        self,
        file_name: str,
        drop_term_list: List[str],
        local_param_dict: Dict[str, Any],
        base_project_dir: str,
        param_space_index: int,
        additional_naming: Optional[str] = None,
        test_sample_n: int = 0,
        column_sample_n: int = 0,
        time_series_mode: bool = False,
        model_class_dict: Optional[Dict[str, bool]] = None,
        outcome_var_override: Optional[str] = None,
    ):
        """Initializes the data pipeline object.

        This method reads data, applies various cleaning and feature engineering
        steps based on the provided parameters, and splits the data into
        training and testing sets.

        Args:
            file_name (str): The path to the input CSV file.
            drop_term_list (List[str]): A list of substrings to identify columns
                to drop.
            local_param_dict (Dict[str, Any]): A dictionary of parameters for this
                specific pipeline run.
            base_project_dir (str): The root directory for the project.
            param_space_index (int): The index of the current parameter space
                permutation.
            additional_naming (Optional[str], optional): Additional string to
                append to log folder names. Defaults to None.
            test_sample_n (int, optional): The number of rows to sample from the
                dataset for testing. Defaults to 0 (no sampling).
            column_sample_n (int, optional): The number of columns to sample.
                Defaults to 0 (no sampling).
            time_series_mode (bool, optional): Flag to enable time-series specific
                data processing. Defaults to False.
            model_class_dict (Optional[Dict[str, bool]], optional): A dictionary
                specifying which model classes to include. Defaults to None.
            outcome_var_override (Optional[str], optional): A specific outcome
                variable name to use, overriding the one from `local_param_dict`.
                Defaults to None.
        """

        self.base_project_dir = base_project_dir

        self.additional_naming = additional_naming

        self.local_param_dict = local_param_dict

        self.global_params = global_parameters

        self.verbose = self.global_params.verbose

        self.param_space_index = param_space_index

        self.time_series_mode = time_series_mode

        self.model_class_dict = model_class_dict

        if self.verbose >= 1:
            print(f"Starting... {self.local_param_dict}")

        self.logging_paths_obj = log_folder(
            local_param_dict=local_param_dict,
            additional_naming=additional_naming,
            base_project_dir=base_project_dir,
        )

        if test_sample_n > 0 or column_sample_n > 0:
            self.df = read_in.read_sample(file_name, test_sample_n, column_sample_n).raw_input_data
        else:
            self.df = read_in.read(file_name, use_polars=True).raw_input_data

        self.all_df_columns = list(self.df.columns)

        self.orignal_feature_names = self.all_df_columns.copy()

        self.pertubation_columns, self.drop_list = get_pertubation_columns(
            all_df_columns=self.all_df_columns,
            local_param_dict=local_param_dict,
            drop_term_list=drop_term_list,
        )
        if outcome_var_override is None:
            self.outcome_variable = f'outcome_var_{local_param_dict.get("outcome_var_n")}'
        
        else:
            print("outcome_var_override:", outcome_var_override)
            print("setting outcome var to:", outcome_var_override)
            self.outcome_variable = outcome_var_override
            
            # get list of variables with substring "outcome_var"
            outcome_vars = [col for col in self.df.columns if "outcome_var" in col]
            print("outcome_vars:", len(outcome_vars))
            
            #remove outcome_var_override from list
            
            outcome_vars.remove(outcome_var_override)
            
            # add additional outcome variables to drop list
            
            self.drop_list.extend(outcome_vars)
            
            
            
        

        print(
            f"Using {len(self.pertubation_columns)}/{len(self.all_df_columns)} columns for {self.outcome_variable} outcome"
        )

        list_2 = self.df.columns
        list_1 = self.pertubation_columns.copy()

        difference_list = list(set(list_2) - set(list_1))
        print(f"Omitting {len(difference_list)} :...")
        print(f"{difference_list[0:5]}...")

        self.drop_list = handle_correlation_matrix(
            local_param_dict=local_param_dict, drop_list=self.drop_list, df=self.df
        )

        self.drop_list = handle_percent_missing(
            local_param_dict=local_param_dict,
            all_df_columns=self.all_df_columns,
            drop_list=self.drop_list,
            file_name=file_name,
        )

        self.drop_list = handle_outcome_list(
            drop_list=self.drop_list, outcome_variable=self.outcome_variable
        )
        
        self.drop_list = remove_constant_columns(
            X=self.df, drop_list=self.drop_list, verbose=self.verbose)

        self.final_column_list = [
            col
            for col in self.pertubation_columns
            if (col not in self.drop_list and col in self.df.columns)
        ]
        # Add safety mechanism to retain minimum features
        min_required_features = 5  # Set your minimum threshold
        core_protected_columns = ['age', 'male', 'client_idcode']  # Columns to protect

        if not self.final_column_list:
            print("WARNING: All features pruned! Activating safety retention...")
            
            # Try to keep protected columns first
            safety_columns = [col for col in core_protected_columns 
                            if col in self.df.columns and col in self.pertubation_columns]
            
            # If no protected columns, use first available columns
            if not safety_columns:
                safety_columns = [col for col in self.pertubation_columns 
                                if col in self.df.columns][:min_required_features]
            
            # Update final columns and drop list
            self.final_column_list = safety_columns
            # Also update the main drop list to prevent re-pruning
            self.drop_list = [col for col in self.drop_list if col not in self.final_column_list]
            
            print(f"Retaining minimum features: {self.final_column_list}")
            
            # Re-filter final_column_list to be absolutely sure
            self.final_column_list = [col for col in self.pertubation_columns if col not in self.drop_list and col in self.df.columns]


            # Add two random features if list still empty
            if not self.final_column_list:
                print("Warning no feature columns retained, selecting two at random")
                self.final_column_list.append(random.choice(self.orignal_feature_names))
                self.final_column_list.append(random.choice(self.orignal_feature_names))

        # Ensure we still have at least 1 feature
        if not self.final_column_list:
            raise ValueError("CRITICAL: Unable to retain any features despite safety measures")

        if not self.final_column_list:
            raise ValueError("All features pruned. No columns remaining in final_column_list.")

        if self.time_series_mode:
            # Re add client_idcode
            self.final_column_list.insert(0, "client_idcode")

        self.X = self.df[self.final_column_list].copy()

        self.X = clean_up_class().handle_duplicated_columns(self.X)

        clean_up_class().screen_non_float_types(self.X)

        self.y = self.df[self.outcome_variable].copy()

        clean_up_class().handle_column_names(self.X)

        scale = self.local_param_dict.get("scale")

        if scale:
            try:
                self.X = data_scale_methods().standard_scale_method(self.X)
            except Exception as e:
                print(e)
                print("Exception scaling data, continuing...")
                print(self.X.shape)
                print(self.X.head())
        if self.verbose >= 1:
            print(
                f"len final droplist: {len(self.drop_list)} \ {len(list(self.df.columns))}"
            )
            # print('\n'.join(map(str, self.drop_list[0:5])))

        print("------------------------")

        # print("LGBM column name fix")

        # Remove special characters and spaces from column names
        # self.X.columns = self.X.columns.str.replace('[^\w\s]', '').str.replace(' ', '_')

        # Convert column names to lowercase
        # self.X.columns = self.X.columns.str.lower()

        #         # Ensure unique column names (in case there are duplicates)
        #         self.X.columns = pd.io.common.dedupe_nans(self.X.columns)

        # self.X = self.X.rename(columns = lambda x:re.sub('[^A-Za-z0-9]+', '', x))

        if self.time_series_mode:
            try:
                from ml_grid.util.time_series_helper import (
                    convert_Xy_to_time_series,
                    max_client_idcode_sequence_length,
                )
            except (ImportError, ModuleNotFoundError):
                print("\n--- WARNING: Time-series libraries not found. ---")
                print(
                    "To run in time-series mode, please install the required dependencies:"
                )
                print(
                    "1. Activate the correct virtual environment: source ml_grid_ts_env/bin/activate"
                )
                print("2. If not installed, run: ./install_ts.sh (or install_ts.bat on Windows)")
                print("-----------------------------------------------------\n")
                raise

            if self.verbose >= 1:
                print("pre func")
                display(self.X)

            max_seq_length = max_client_idcode_sequence_length(self.df)

            if self.verbose >= 1:
                print("time_series_mode", "convert_df_to_time_series")
                print(self.X.shape)

            self.X, self.y = convert_Xy_to_time_series(self.X, self.y, max_seq_length)
            if self.verbose >= 1:
                print(self.X.shape)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.X_test_orig,
            self.y_test_orig,
        ) = get_data_split(X=self.X, y=self.y, local_param_dict=self.local_param_dict)
        
        # Handle columns made constant by splitting
        self.X_train, self.X_test, self.X_test_orig = remove_constant_columns_with_debug(
            self.X_train,
            self.X_test,
            self.X_test_orig,
            verbosity=self.verbose
        )

        target_n_features = self.local_param_dict.get("feature_n")

        if target_n_features != 100:

            target_n_features_eval = int(
                (target_n_features / 100) * self.X_train.shape[1]
            )

            # Ensure at least one feature is selected. The previous logic here
            # was incorrect and disabled feature selection entirely.
            target_n_features_eval = max(1, target_n_features_eval)

            print(
                f"Pre target_n_features {target_n_features}% reduction {target_n_features_eval}/{self.X_train.shape[1]}"
            )
            try:

                fim = feature_importance_methods()
                self.X_train, self.X_test, self.X_test_orig = (
                    fim.handle_feature_importance_methods(
                        target_n_features_eval,
                        X_train=self.X_train,
                        X_test=self.X_test,
                        y_train=self.y_train,
                        X_test_orig=self.X_test_orig,
                        ml_grid_object=self
                    )
                )
                if self.X_train.shape[1] == 0:
                    raise ValueError("Feature importance selection removed all features.")

            except Exception as e:
                print("failed target_n_features", e)

        if self.verbose >= 2:
            print(
                f"Data Split Information:\n"
                f"Number of rows in self.X_train: {len(self.X_train)}, Columns: {self.X_train.shape[1]}\n"
                f"Number of rows in self.X_test: {len(self.X_test)}, Columns: {self.X_test.shape[1]}\n"
                f"Number of rows in self.y_train: {len(self.y_train)}\n"
                f"Number of rows in self.y_test: {len(self.y_test)}\n"
                f"Number of rows in self.X_test_orig: {len(self.X_test_orig)}, Columns: {self.X_test_orig.shape[1]}\n"
                f"Number of rows in self.y_test_orig: {len(self.y_test_orig)}"
            )

        if self.verbose >= 3:

            plot_pie_chart_with_counts(self.X_train, self.X_test, self.X_test_orig)

        if time_series_mode:
            if self.verbose >= 2:
                print("data>>", "get_model_class_list_ts")
            try:
                from ml_grid.pipeline.model_class_list_ts import (
                    get_model_class_list_ts,
                )
            except (ImportError, ModuleNotFoundError):
                print("\n--- WARNING: Time-series libraries not found. ---")
                print(
                    "To run in time-series mode, please install the required dependencies:"
                )
                print(
                    "1. Activate the correct virtual environment: source ml_grid_ts_env/bin/activate"
                )
                print("2. If not installed, run: ./install_ts.sh (or install_ts.bat on Windows)")
                print("-----------------------------------------------------\n")
                raise
            self.model_class_list = get_model_class_list_ts(self)

        else:
            if self.verbose >= 2:
                print("data>>", "get_model_class_list")
            if model_class_dict is not None:
                self.model_class_dict = model_class_dict
            
            from ml_grid.pipeline.model_class_list import get_model_class_list

            self.model_class_list = get_model_class_list(self)

        if isinstance(self.X_train, pd.DataFrame) and self.X_train.empty:
            raise ValueError("-- end data pipeline-- Input data X_train is an empty DataFrame -- end data pipeline--")