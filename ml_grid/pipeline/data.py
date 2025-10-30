import random
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from IPython.display import display
from pandas.testing import assert_index_equal
from sklearn.exceptions import ConvergenceWarning

from ml_grid.pipeline import read_in
from ml_grid.pipeline.column_names import get_pertubation_columns
from ml_grid.pipeline.data_clean_up import clean_up_class
from ml_grid.pipeline.data_constant_columns import (
    remove_constant_columns_with_debug,
)
from ml_grid.pipeline.data_correlation_matrix import handle_correlation_matrix
from ml_grid.pipeline.data_feature_importance_methods import feature_importance_methods
from ml_grid.pipeline.data_outcome_list import handle_outcome_list
from ml_grid.pipeline.data_plot_split import (
    plot_pie_chart_with_counts,
)  # This import is not used in the provided code, but kept as it's not the focus of this fix.
from ml_grid.pipeline.data_train_test_split import get_data_split
from ml_grid.pipeline.embeddings import create_embedding_pipeline
from ml_grid.util.global_params import global_parameters
from ml_grid.util.logger_setup import setup_logger

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.preprocessing import (
    StandardScaler,
)  # Added explicit import for StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


class NoFeaturesError(Exception):
    """Custom exception raised when no features are left after processing."""

    pass


class pipe:
    """Represents a single data processing pipeline permutation.

    This class reads data, applies a series of cleaning and feature selection
    steps based on a dictionary of parameters, and prepares the data for model
    training and evaluation. The resulting object holds the processed data and
    can be passed to classifier methods.
    """

    additional_naming: Optional[str]
    """An optional string to append to log folder names for better identification."""

    local_param_dict: Dict[str, Any]
    """A dictionary of parameters for this specific pipeline run."""

    global_params: global_parameters
    """A reference to the global parameters singleton instance."""

    verbose: int
    """The verbosity level for logging, inherited from global parameters."""

    param_space_index: int
    """The index of the current parameter space permutation being run."""

    time_series_mode: bool
    """A flag indicating if the pipeline is running in time-series mode."""

    model_class_dict: Optional[Dict[str, bool]]
    """A dictionary specifying which model classes to include in the run."""

    df: pd.DataFrame
    """The raw input DataFrame after being read from the source file."""

    all_df_columns: List[str]
    """A list of all column names from the original raw DataFrame."""

    orignal_feature_names: List[str]
    """A copy of the original feature names before any processing."""

    pertubation_columns: List[str]
    """A list of columns selected for inclusion based on `local_param_dict`."""

    drop_list: List[str]
    """A list of columns identified to be dropped due to various cleaning steps."""

    outcome_variable: str
    """The name of the target variable for the current pipeline run."""

    final_column_list: List[str]
    """The final list of feature columns to be used after all filtering."""

    X: pd.DataFrame
    """The feature matrix (DataFrame) after all cleaning and selection steps."""

    y: pd.Series
    """The target variable (Series) corresponding to the feature matrix `X`."""

    X_train: pd.DataFrame
    """The training feature set."""

    X_test: pd.DataFrame
    """The validation/testing feature set."""

    y_train: pd.Series
    """The training target set."""

    y_test: pd.Series
    """The validation/testing target set."""

    X_test_orig: pd.DataFrame
    """The original, held-out test set for final validation."""

    y_test_orig: pd.Series
    """The target variable for the original, held-out test set."""

    model_class_list: List[Any]
    """A list of instantiated model class objects to be evaluated in this run."""

    feature_transformation_log: pd.DataFrame
    """A DataFrame that logs the changes to the feature set at each pipeline step."""

    def _log_feature_transformation(
        self, step_name: str, before_count: int, after_count: int, description: str
    ):
        """Helper function to log feature transformation steps."""
        if self.verbose >= 1:
            self._feature_log_list.append(
                {
                    "step": step_name,
                    "features_before": before_count,
                    "features_after": after_count,
                    "features_changed": before_count - after_count,
                    "description": description,
                }
            )

    def _assert_index_alignment(
        self, df1: pd.DataFrame, df2: pd.Series, step_name: str
    ):
        """Helper function to assert that DataFrame and Series indices are equal."""
        try:
            assert_index_equal(df1.index, df2.index)
            self.logger.debug(f"Index alignment PASSED at: {step_name}")
        except AssertionError:
            self.logger.error(f"Index alignment FAILED at: {step_name}")
            raise

    def _setup_pipeline(self, experiment_dir: str, redirect_stdout: bool):
        """Initializes logger and sets up basic attributes."""
        self.logger = setup_logger(
            experiment_dir=experiment_dir,
            param_space_index=self.param_space_index,
            verbose=self.verbose,
            redirect_stdout=redirect_stdout,
        )
        self.logger.info("Logger setup complete.")
        self._feature_log_list = []
        self.logger.info(
            f"Starting pipeline run for param space index {self.param_space_index}"
        )
        self.logger.info(f"Parameters: {self.local_param_dict}")

    def _load_data(self, file_name: str, test_sample_n: int, column_sample_n: int):
        """Loads data from the source file."""
        if test_sample_n > 0 or column_sample_n > 0:
            self.df = read_in.read_sample(
                file_name, test_sample_n, column_sample_n
            ).raw_input_data
        else:
            self.df = read_in.read(file_name, use_polars=True).raw_input_data

        self.all_df_columns = list(self.df.columns)
        self.orignal_feature_names = self.all_df_columns.copy()
        self._log_feature_transformation(
            "Initial Load",
            len(self.all_df_columns),
            len(self.all_df_columns),
            "Initial data loaded.",
        )

    def __init__(
        self,
        file_name: str,
        drop_term_list: List[str],
        experiment_dir: str,
        base_project_dir: str,
        local_param_dict: Dict[str, Any],
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
            base_project_dir (str): The root directory for the project where
                logs and models will be saved.
            experiment_dir (str): The path to the parent directory for this group
                of experimental runs.
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

        self.additional_naming = additional_naming

        self.base_project_dir = base_project_dir

        self.experiment_dir = experiment_dir

        self.local_param_dict = local_param_dict

        self.global_params = global_parameters

        self.verbose = self.global_params.verbose

        self.param_space_index = param_space_index

        self.time_series_mode = time_series_mode

        self.model_class_dict = model_class_dict

        redirect_stdout = not self.global_params.bayessearch
        self._setup_pipeline(experiment_dir, redirect_stdout)

        pipeline_error = None
        try:
            self._load_data(file_name, test_sample_n, column_sample_n)
            self._initial_feature_selection(
                local_param_dict, drop_term_list, outcome_var_override
            )
            self._apply_safety_net()
            self._create_xy()
            self._handle_time_series_conversion()
            self._split_data()
            self._post_split_cleaning()
            self._scale_features()
            self._select_features_by_importance()
            self._apply_embeddings()
            self._finalize_pipeline()
        except Exception as e:
            pipeline_error = e
            # Re-raise the exception after the finally block has executed
            raise
        finally:
            # This block ensures the feature log is always processed and displayed,
            # even if an error occurs during the pipeline execution.
            self._compile_and_log_feature_transformations(
                error_occurred=pipeline_error is not None
            )
            if pipeline_error:
                self.logger.error("Data pipeline processing HALTED due to an error.")
            else:
                self.logger.info("Data pipeline processing complete.")

    def _initial_feature_selection(
        self, local_param_dict, drop_term_list, outcome_var_override
    ):
        """Performs initial feature selection based on toggles and determines outcome variable."""
        self.pertubation_columns, self.drop_list = get_pertubation_columns(
            all_df_columns=self.all_df_columns,
            local_param_dict=local_param_dict,
            drop_term_list=drop_term_list,
        )
        if outcome_var_override is None:
            self.outcome_variable = (
                f'outcome_var_{local_param_dict.get("outcome_var_n")}'
            )
        else:
            self.logger.info(f"outcome_var_override provided: {outcome_var_override}")
            self.logger.info(f"Setting outcome var to: {outcome_var_override}")
            self.outcome_variable = outcome_var_override

        self._log_feature_transformation(
            "Feature Selection (Toggles)",
            len(self.all_df_columns),
            len(self.pertubation_columns),
            "Selected columns based on feature toggles in config.",
        )

        self.logger.info(
            f"Using {len(self.pertubation_columns)}/{len(self.all_df_columns)} columns for {self.outcome_variable} outcome"
        )

        # Log omitted columns for debugging
        difference_list = list(set(self.df.columns) - set(self.pertubation_columns))
        self.logger.info(
            f"Omitting {len(difference_list)} columns based on feature toggles."
        )
        self.logger.debug(f"Sample of omitted columns: {difference_list[0:5]}...")

        # Consolidate dropping of other outcome variables
        features_before = len(self.pertubation_columns)
        self.drop_list = handle_outcome_list(
            drop_list=self.drop_list, outcome_variable=self.outcome_variable
        )
        # Recalculate features to be kept after updating the drop list
        features_after = [
            col for col in self.pertubation_columns if col not in self.drop_list
        ]
        self._log_feature_transformation(
            "Drop Other Outcomes",
            features_before,
            len(features_after),
            "Removed other potential outcome variables from feature set.",
        )

        # Combine all columns to be kept
        current_features = [
            col
            for col in self.pertubation_columns
            if col not in self.drop_list and col != self.outcome_variable
        ]

        self.final_column_list = current_features

    def _apply_safety_net(self):
        """Retains a minimal set of features if all have been pruned."""
        if not self.final_column_list:
            self.logger.warning(
                "All features were pruned. Activating safety retention mechanism."
            )

            # Define core columns to try and protect
            core_protected_columns = ["age", "male", "client_idcode"]
            min_features = 2

            # 1. Try to retain core protected columns
            retained_cols = [
                col
                for col in core_protected_columns
                if col in self.pertubation_columns and col in self.df.columns
            ]

            # 2. If no core columns, try to retain any of the original perturbed columns
            if not retained_cols:
                retained_cols = [
                    col for col in self.pertubation_columns if col in self.df.columns
                ][:min_features]

            # 3. As a last resort, pick random columns from the original features
            if not retained_cols:
                self.logger.warning("Last resort: Selecting random features.")
                available_features = [
                    col
                    for col in self.orignal_feature_names
                    if col != self.outcome_variable and col in self.df.columns
                ]
                if len(available_features) >= min_features:
                    retained_cols = random.sample(available_features, min_features)
                elif available_features:
                    retained_cols = available_features

            self.final_column_list = retained_cols
            self.logger.info(f"Retained minimum features: {self.final_column_list}")
            self._log_feature_transformation(
                "Safety Net",
                0,
                len(self.final_column_list),
                "All features were pruned; safety net retained a minimal set.",
            )

        # Final check to ensure we have at least one feature.
        # If even the safety net fails, we cannot proceed.
        if not self.final_column_list:
            raise NoFeaturesError(
                "CRITICAL: Unable to retain any features despite safety measures. Halting pipeline."
            )

    def _create_xy(self):
        """Creates the feature matrix X and target vector y."""
        if self.time_series_mode:
            self.final_column_list.insert(0, "client_idcode")

        self.X = self.df[self.final_column_list]

        self._assert_index_alignment(
            self.X, self.df[self.outcome_variable], "Initial X creation"
        )

        self.y = self.df[self.outcome_variable]

        self._assert_index_alignment(self.X, self.y, "Initial y creation")

        # --- CRITICAL FIX for indexing errors ---
        # Reset index here to ensure all downstream functions (splitting, CV)
        # receive data with a clean, standard 0-based integer index.
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        self._assert_index_alignment(self.X, self.y, "After initial reset_index")

        self.logger.info(
            f"len final droplist: {len(self.drop_list)} / {len(list(self.df.columns))}"
        )
        # print('\n'.join(map(str, self.drop_list[0:5])))

        # Remove special characters and spaces from column names
        # self.X.columns = self.X.columns.str.replace('[^\w\s]', '').str.replace(' ', '_')

        # Convert column names to lowercase
        # self.X.columns = self.X.columns.str.lower()

    def _handle_time_series_conversion(self):
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
                self.logger.error("\n--- WARNING: Time-series libraries not found. ---")
                self.logger.error(
                    "To run in time-series mode, please install the required dependencies:"
                )
                self.logger.error(
                    "1. Activate the correct virtual environment: source ml_grid_ts_env/bin/activate"
                )
                self.logger.error(
                    "2. If not installed, run: ./install_ts.sh (or install_ts.bat on Windows)"
                )
                raise

            self.logger.debug("Preparing for time-series conversion.")
            # display(self.X) # display() will still work in notebooks

            max_seq_length = max_client_idcode_sequence_length(self.df)
            self.logger.info(f"Max sequence length for time-series: {max_seq_length}")

            self.logger.info("time_series_mode: convert_df_to_time_series")
            self.logger.info(self.X.shape)

            self.X, self.y = convert_Xy_to_time_series(self.X, self.y, max_seq_length)
            if (
                not self.final_column_list
            ):  # This condition seems odd, but preserving logic
                self.logger.info(self.X.shape)

    def _split_data(self):
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.X_test_orig,
            self.y_test_orig,
        ) = get_data_split(X=self.X, y=self.y, local_param_dict=self.local_param_dict)

        self._assert_index_alignment(self.X_train, self.y_train, "After get_data_split")

        # --- CRITICAL FIX: Reset all indices immediately after splitting ---
        # This ensures all downstream processing (constant removal, feature selection, embedding)
        # operates on data with clean, aligned, 0-based integer indices.
        self.X_train.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        self.X_test_orig.reset_index(drop=True, inplace=True)
        self.y_test_orig.reset_index(drop=True, inplace=True)
        self._assert_index_alignment(
            self.X_train, self.y_train, "After master reset_index"
        )

    def _post_split_cleaning(self):
        """Applies cleaning steps post-split to prevent data leakage."""
        # Clean column names *before* dropping operations to ensure stable column order.
        cleanup = clean_up_class()
        cleanup.screen_non_float_types(self.X_train)
        cleanup.handle_column_names(self.X_train)
        # Apply the exact same column names from X_train to the test sets for consistency.
        self.X_test.columns = self.X_train.columns
        self.X_test_orig.columns = self.X_train.columns
        self._assert_index_alignment(
            self.X_train, self.y_train, "After cleanup and column renaming"
        )

        # 2. Handle columns with high percentage of missing values based on X_train
        features_before = self.X_train.shape[1]
        # We need to re-implement the logic here as handle_percent_missing was not designed for this.
        # A simpler approach is to calculate missing % on X_train and drop.
        percent_missing_threshold = self.local_param_dict.get("percent_missing", 100)
        if percent_missing_threshold < 100:
            missing_perc = self.X_train.isnull().sum() / len(self.X_train) * 100
            missing_drop_list = missing_perc[
                missing_perc > percent_missing_threshold
            ].index.tolist()

            self.X_train.drop(columns=missing_drop_list, inplace=True, errors="ignore")
            self.X_test.drop(columns=missing_drop_list, inplace=True, errors="ignore")
            self.X_test_orig.drop(
                columns=missing_drop_list, inplace=True, errors="ignore"
            )
            self._log_feature_transformation(
                "Drop Missing (Post-Split)",
                features_before,
                self.X_train.shape[1],
                f"Dropped columns with > {percent_missing_threshold}% missing values based on X_train.",
            )

        # 1. Handle columns that are constant *within* the training set
        features_before = self.X_train.shape[1]
        self.X_train, self.X_test, self.X_test_orig = (
            remove_constant_columns_with_debug(
                self.X_train,
                self.X_test,
                self.X_test_orig,
                verbosity=self.verbose,
            )
        )
        self._log_feature_transformation(
            "Drop Post-Split Constants",
            features_before,
            self.X_train.shape[1],
            "Removed columns that became constant after train/test split.",
        )

        # 2. Handle highly correlated features based on X_train (AFTER constant removal)
        features_before = self.X_train.shape[1]
        corr_drop_list = handle_correlation_matrix(
            local_param_dict=self.local_param_dict, drop_list=[], df=self.X_train
        )
        if corr_drop_list:
            self.X_train.drop(columns=corr_drop_list, inplace=True, errors="ignore")
            # Ensure test sets have the same columns as X_train after correlation removal
            self.X_test = self.X_test[self.X_train.columns]
            self.X_test_orig = self.X_test_orig[self.X_train.columns]
        self._log_feature_transformation(
            "Drop Correlated (Post-Split)",
            features_before,
            self.X_train.shape[1],
            f"Dropped columns with correlation > {self.local_param_dict.get('corr')} based on X_train.",
        )

        # Handle duplicated columns (after other removals)
        features_before = self.X_train.shape[1]
        original_cols = self.X_train.columns.tolist()
        self.X_train = clean_up_class().handle_duplicated_columns(self.X_train)
        dropped_cols = list(set(original_cols) - set(self.X_train.columns))
        if dropped_cols:
            self.X_test.drop(columns=dropped_cols, inplace=True, errors="ignore")
            self.X_test_orig.drop(columns=dropped_cols, inplace=True, errors="ignore")
        self._log_feature_transformation(
            "Drop Duplicated Columns",
            features_before,
            self.X_train.shape[1],
            "Removed duplicated columns based on X_train.",
        )

        self.final_column_list = self.X_train.columns.tolist()

        self.logger.info(
            f"Shape of X_train after post-split cleaning: {self.X_train.shape}"
        )

        # Final check after all post-split cleaning steps.
        if self.X_train.shape[1] == 0:
            raise NoFeaturesError(
                "All feature columns were removed after data splitting and cleaning. Consider adjusting feature selection or data cleaning parameters."
            )

    def _scale_features(self):
        """Applies standard scaling to the feature sets."""
        features_before = self.X_train.shape[1]
        scale = self.local_param_dict.get("scale")
        if scale:
            if self.X_train.shape[1] > 0:
                try:
                    scaler = (
                        StandardScaler()
                    )  # Use StandardScaler directly to resolve AttributeError
                    self.X_train = pd.DataFrame(
                        scaler.fit_transform(self.X_train),
                        columns=self.X_train.columns,
                        index=self.X_train.index,
                    )
                    self.X_test = pd.DataFrame(
                        scaler.transform(self.X_test),
                        columns=self.X_test.columns,
                        index=self.X_test.index,
                    )
                    self.X_test_orig = pd.DataFrame(
                        scaler.transform(self.X_test_orig),
                        columns=self.X_test_orig.columns,
                        index=self.X_test_orig.index,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Exception scaling data post-split: {e}", exc_info=True
                    )
                    self.logger.warning("Continuing without scaling.")
            else:
                self.logger.warning(
                    "Skipping scaling because no features are present in X_train."
                )
        self._log_feature_transformation(
            "Standard Scaling",
            features_before,
            self.X_train.shape[1],
            "Applied StandardScaler to numeric features based on X_train.",
        )
        self._assert_index_alignment(self.X_train, self.y_train, "After scaling")

    def _select_features_by_importance(self):
        """Selects features based on importance scores if configured."""
        target_n_features = self.local_param_dict.get("feature_n")

        if target_n_features is not None and target_n_features < 100:
            target_n_features_eval = int(
                (target_n_features / 100) * self.X_train.shape[1]
            )
            # Ensure at least one feature is selected. The previous logic here
            # was incorrect and disabled feature selection entirely.
            target_n_features_eval = max(1, target_n_features_eval)

        if (
            target_n_features is not None
            and target_n_features < 100
            and self.X_train.shape[1] > 1
            and not self.local_param_dict.get("use_embedding", False)
        ):
            features_before = self.X_train.shape[1]

            self.logger.info(
                f"Shape of X_train before feature importance selection: {self.X_train.shape}"
            )

            self.logger.info(
                f"Pre target_n_features {target_n_features}% reduction {target_n_features_eval}/{self.X_train.shape[1]}"
            )
            try:

                fim = feature_importance_methods()
                (
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test,
                    self.X_test_orig,
                ) = fim.handle_feature_importance_methods(
                    target_n_features_eval,
                    X_train=self.X_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    y_train=self.y_train,
                    X_test_orig=self.X_test_orig,
                    ml_grid_object=self,
                )
                self._log_feature_transformation(
                    "Feature Importance",
                    features_before,
                    self.X_train.shape[1],
                    f"Selected top {target_n_features}% features using {fim.feature_method}.",
                )
                self._assert_index_alignment(
                    self.X_train,
                    self.y_train,
                    "After feature selection and y_train reset",
                )

                self.logger.info(
                    f"Shape of X_train after feature importance selection: {self.X_train.shape}"
                )

                if self.X_train.shape[1] == 0:
                    raise ValueError(
                        "Feature importance selection removed all features."
                    )

                # Safeguard: Ensure X_train is not empty after feature selection
                if self.X_train.shape[1] == 0:
                    raise ValueError(
                        "All features were removed by the feature importance selection method. X_train is empty."
                    )

            except Exception as e:
                self.logger.error(
                    f"Feature importance selection failed: {e}", exc_info=True
                )
        self._assert_index_alignment(
            self.X_train, self.y_train, "After feature selection block"
        )

    def _apply_embeddings(self):
        """Applies feature embedding/dimensionality reduction if configured."""
        if self.local_param_dict.get("use_embedding", False):
            features_before = self.X_train.shape[1]
            self.logger.info("Applying embeddings...")

            # Safeguard: Some embedding methods require at least 2 features.
            if self.X_train.shape[1] < 2:
                self.logger.warning(
                    f"Skipping embedding: Only {self.X_train.shape[1]} feature(s) available, but embedding requires at least 2."
                )
                # The pipeline will continue without applying embeddings.
            else:

                embedding_method = self.local_param_dict.get("embedding_method", "pca")
                embedding_dim = self.local_param_dict.get("embedding_dim", 64)
                scale_before_embedding = self.local_param_dict.get(
                    "scale_features_before_embedding", True
                )

                self.logger.info(f"  Embedding Method: {embedding_method}")
                self.logger.info(f"  Original features: {self.X_train.shape[1]}")
                self.logger.info(f"  Target embedding dimensions: {embedding_dim}")
                self.logger.info(f"  Scale before embedding: {scale_before_embedding}")

                # Safeguard: n_components must be less than n_features.
                if embedding_dim >= self.X_train.shape[1]:
                    # Adjust embedding_dim to be the number of available features.
                    embedding_dim = self.X_train.shape[1]
                    self.logger.warning(
                        f"  embedding_dim >= n_features. Adjusting to {embedding_dim}."
                    )

                embedding_pipeline = create_embedding_pipeline(
                    method=embedding_method,
                    n_components=embedding_dim,
                    scale=scale_before_embedding,
                )

                # Fit on train and transform all splits
                # Check if the method is supervised to pass y_train
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                from sklearn.feature_selection import SelectKBest

                from ml_grid.pipeline.embeddings import get_explained_variance

                embed_step = embedding_pipeline.named_steps["embed"]
                if isinstance(embed_step, (LinearDiscriminantAnalysis, SelectKBest)):
                    self.logger.debug(
                        "  Supervised embedding method detected, passing y_train."
                    )
                    self.X_train = pd.DataFrame(
                        embedding_pipeline.fit_transform(
                            self.X_train, self.y_train.values
                        ),
                        index=self.X_train.index,
                        columns=[f"embed_{i}" for i in range(embedding_dim)],
                    )
                else:
                    self.X_train = pd.DataFrame(
                        embedding_pipeline.fit_transform(self.X_train),
                        index=self.X_train.index,
                        columns=[f"embed_{i}" for i in range(embedding_dim)],
                    )

                self.X_test = pd.DataFrame(
                    embedding_pipeline.transform(self.X_test),
                    index=self.X_test.index,
                    columns=[f"embed_{i}" for i in range(embedding_dim)],
                )
                self.X_test_orig = pd.DataFrame(
                    embedding_pipeline.transform(self.X_test_orig),
                    index=self.X_test_orig.index,
                    columns=[f"embed_{i}" for i in range(embedding_dim)],
                )

                # The main self.X should also be updated for consistency, using the training data's embedding

                # CRITICAL: Re-run constant column removal after embedding, as the process
                # itself can create constant columns (e.g., PCA components with zero variance).
                features_before_post_embed_const = self.X_train.shape[1]
                self.X_train, self.X_test, self.X_test_orig = (
                    remove_constant_columns_with_debug(
                        self.X_train,
                        self.X_test,
                        self.X_test_orig,
                        verbosity=self.verbose,
                    )
                )
                self._log_feature_transformation(
                    "Drop Post-Embedding Constants",
                    features_before_post_embed_const,
                    self.X_train.shape[1],
                    "Removed constant columns created by the embedding process.",
                )

                self._log_feature_transformation(
                    "Embedding",
                    features_before,
                    self.X_train.shape[1],
                    f"Applied {embedding_method} to reduce features to {embedding_dim} dimensions.",
                )
                self.X = self.X_train.copy()
                self._assert_index_alignment(
                    self.X_train, self.y_train, "After embedding"
                )

                self.logger.info(
                    f"Shape of X_train after embedding: {self.X_train.shape}"
                )
                self.logger.info(
                    f"Data transformed to {self.X_train.shape[1]} embedding dimensions."
                )
                explained_variance = get_explained_variance(embedding_pipeline)
                if explained_variance is not None:
                    self.logger.info(
                        f"  Total explained variance by {embedding_dim} components: {explained_variance.sum():.2%}"
                    )

    def _finalize_pipeline(self):
        """Final logging, checks, and model list generation."""

        if self.verbose >= 2:
            self.logger.debug(
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

        if self.time_series_mode:
            self.logger.debug("Getting time-series model class list")
            try:
                from ml_grid.pipeline.model_class_list_ts import (
                    get_model_class_list_ts,
                )
            except (ImportError, ModuleNotFoundError):
                self.logger.error("\n--- WARNING: Time-series libraries not found. ---")
                self.logger.error(
                    "To run in time-series mode, please install the required dependencies:"
                )
                self.logger.error(
                    "1. Activate the correct virtual environment: source ml_grid_ts_env/bin/activate"
                )
                self.logger.error(
                    "2. If not installed, run: ./install_ts.sh (or install_ts.bat on Windows)"
                )
                raise
            self.model_class_list = get_model_class_list_ts(self)

        else:
            self.logger.debug("Getting standard model class list")
            if self.model_class_dict is not None:
                self.model_class_dict = self.model_class_dict

            from ml_grid.pipeline.model_class_list import get_model_class_list

            self.model_class_list = get_model_class_list(self)

        if isinstance(self.X_train, pd.DataFrame) and self.X_train.empty:
            raise ValueError(
                "-- end data pipeline-- Input data X_train is an empty DataFrame. "
                "This is likely due to aggressive feature selection or data cleaning."
            )

        # Final definitive assertion before exiting the data pipeline.
        # This ensures that the X_train and y_train that will be passed to the
        # model training steps are perfectly aligned.
        try:
            assert_index_equal(self.X_train.index, self.y_train.index)
            self.logger.info(
                "Final data alignment check PASSED. X_train and y_train indices are identical."
            )
        except AssertionError:
            self.logger.error(
                "CRITICAL: Final data alignment check FAILED. X_train and y_train indices are NOT identical."
            )
            raise

    def _compile_and_log_feature_transformations(self, error_occurred: bool = False):
        """Compiles the feature transformation log and displays it."""
        # Ensure y_train is a pandas Series for consistency before exiting.
        if hasattr(self, "y_train") and not isinstance(self.y_train, pd.Series):
            self.y_train = pd.Series(self.y_train, index=self.X_train.index)

        # Finalize the feature transformation log
        if self._feature_log_list:
            self.feature_transformation_log = pd.DataFrame(self._feature_log_list)
            log_string = self.feature_transformation_log.to_string()

            if error_occurred:
                # If an error happened, always log the transformation table for debugging.
                self.logger.error(
                    "\n--- Feature Transformation Log (at time of error) ---\n"
                    + log_string
                )
            elif self.verbose >= 1:
                # Otherwise, log it based on verbosity.
                self.logger.info("\n--- Feature Transformation Log ---\n" + log_string)
                display(self.feature_transformation_log)
                self.logger.info("--------------------------------\n")
