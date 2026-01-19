import time
import logging
import multiprocessing
import joblib
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from IPython.display import clear_output
from scikeras.wrappers import KerasClassifier
import sklearn
from sklearn import metrics
from pandas.testing import assert_index_equal
from xgboost.core import XGBoostError
from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier
from ml_grid.model_classes.H2OGBMClassifier import H2OGBMClassifier
from ml_grid.model_classes.H2ODRFClassifier import H2ODRFClassifier
from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier
from ml_grid.model_classes.H2ODeepLearningClassifier import H2ODeepLearningClassifier
from ml_grid.model_classes.H2OGLMClassifier import H2OGLMClassifier
from ml_grid.model_classes.H2ONaiveBayesClassifier import H2ONaiveBayesClassifier
from ml_grid.model_classes.H2ORuleFitClassifier import H2ORuleFitClassifier
from ml_grid.model_classes.H2OXGBoostClassifier import H2OXGBoostClassifier
from ml_grid.model_classes.H2OStackedEnsembleClassifier import (
    H2OStackedEnsembleClassifier,
)
from ml_grid.model_classes.NeuralNetworkKerasClassifier import NeuralNetworkClassifier

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import *
from sklearn.model_selection import (
    ParameterGrid,
    RepeatedKFold,
    KFold,
    cross_validate,
)

from ml_grid.model_classes.keras_classifier_class import KerasClassifierClass
from ml_grid.pipeline.hyperparameter_search import HyperparameterSearch
from ml_grid.util.debug_print_statements import debug_print_statements_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class
from ml_grid.util.validate_parameters import validate_parameters_helper
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ml_grid.util.bayes_utils import is_skopt_space
from skopt.space import Categorical

# Global flag to ensure TensorFlow/GPU setup runs only once per process
_TF_INITIALIZED = False

# Define H2O model types at module level for reuse
H2O_MODEL_TYPES = (
    H2OAutoMLClassifier,
    H2OGBMClassifier,
    H2ODRFClassifier,
    H2OGAMClassifier,
    H2ODeepLearningClassifier,
    H2OGLMClassifier,
    H2ONaiveBayesClassifier,
    H2ORuleFitClassifier,
    H2OXGBoostClassifier,
    H2OStackedEnsembleClassifier,
)

# --- OPTIMIZATION: Disable TF Traceback Filtering Globally ---
# This reduces overhead in Keras model building (sequential.py, tracking.py)
# seen in profiling (traceback_utils.py taking ~60s).
try:
    tf.debugging.disable_traceback_filtering()
except (AttributeError, ImportError):
    pass


class grid_search_crossvalidate:

    def __init__(
        self,
        algorithm_implementation: Any,
        parameter_space: Union[Dict, List[Dict]],
        method_name: str,
        ml_grid_object: Any,
        sub_sample_parameter_val: int = 100,
        project_score_save_class_instance: Optional[project_score_save_class] = None,
    ):
        """Initializes and runs a cross-validated hyperparameter search.

        This class takes a given algorithm, its parameter space, and data from
        the main pipeline object to perform either a grid search, randomized
        search, or Bayesian search for the best hyperparameters. It then logs
        the results.

        Args:
            algorithm_implementation (Any): The scikit-learn compatible estimator
                instance.
            parameter_space (Union[Dict, List[Dict]]): The dictionary or list of
                dictionaries defining the hyperparameter search space.
            method_name (str): The name of the algorithm method.
            ml_grid_object (Any): The main pipeline object containing all data
                (X_train, y_train, etc.) and parameters for the current
                iteration.
            sub_sample_parameter_val (int, optional): A value used to limit
                the number of iterations in a randomized search. Defaults to 100.
            project_score_save_class_instance (Optional[project_score_save_class], optional):
                An instance of the score saving class. Defaults to None.
        """
        # Set each warning filter individually for robustness
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.logger = logging.getLogger("ml_grid")

        self.global_params = global_parameters

        self.verbose = self.global_params.verbose

        if self.verbose < 8:
            self.logger.debug("Clearing output.")
            clear_output(wait=True)

        self.project_score_save_class_instance = project_score_save_class_instance

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        random_grid_search = self.global_params.random_grid_search

        self.sub_sample_parameter_val = sub_sample_parameter_val

        # --- OPTIMIZATION: Detect Nested Parallelism ---
        # If running inside a worker process (daemon), force n_jobs=1 to prevent
        # oversubscription (outer loop parallel * inner loop parallel).
        if multiprocessing.current_process().daemon:
            self.global_params.grid_n_jobs = 1
            grid_n_jobs = 1
        else:
            grid_n_jobs = self.global_params.grid_n_jobs

        # Configure GPU usage and job limits for specific models
        is_gpu_model = (
            "keras" in method_name.lower()
            or "xgb" in method_name.lower()
            or "catboost" in method_name.lower()
            or "neural" in method_name.lower()
        )

        is_h2o_model = isinstance(algorithm_implementation, H2O_MODEL_TYPES)

        global _TF_INITIALIZED
        if is_gpu_model or is_h2o_model:
            grid_n_jobs = 1

            # --- OPTIMIZATION: Disable H2O Progress Bar ---
            # This saves significant time (~95s) spent in progressbar updates
            if is_h2o_model:
                try:
                    import h2o

                    h2o.no_progress()
                except ImportError:
                    pass
                except Exception:
                    pass

            # --- OPTIMIZATION: One-time TF/GPU Setup ---
            if is_gpu_model and not _TF_INITIALIZED:
                try:
                    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
                    if gpu_devices:
                        for device in gpu_devices:
                            try:
                                tf.config.experimental.set_memory_growth(device, True)
                            except RuntimeError:
                                pass  # Memory growth must be set before GPUs have been initialized
                    else:
                        # Explicitly set CPU as the visible device for TensorFlow to avoid CUDA init errors
                        tf.config.set_visible_devices([], "GPU")
                    
                    tf.config.run_functions_eagerly(False)
                except Exception as e:
                    self.logger.warning(f"Could not configure GPU for TensorFlow: {e}")
                finally:
                    _TF_INITIALIZED = True

        self.metric_list = self.global_params.metric_list

        self.error_raise = self.global_params.error_raise

        if self.verbose >= 3:
            self.logger.info(f"Cross-validating {method_name}")

        self.global_parameters = global_parameters

        self.ml_grid_object_iter = ml_grid_object

        self.X_train = self.ml_grid_object_iter.X_train

        self.y_train = self.ml_grid_object_iter.y_train

        self.X_test = self.ml_grid_object_iter.X_test

        self.y_test = self.ml_grid_object_iter.y_test

        self.X_test_orig = self.ml_grid_object_iter.X_test_orig

        self.y_test_orig = self.ml_grid_object_iter.y_test_orig

        # --- ROBUST DATA TYPE HANDLING ---
        # Ensure X_train is a pandas DataFrame and y_train is a pandas Series
        # with aligned indices. This handles inputs being numpy arrays (from tests)
        # or pandas objects, preventing AttributeError and ensuring consistency.

        # 1. Ensure X_train is a DataFrame.
        if not isinstance(self.X_train, pd.DataFrame):
            self.X_train = pd.DataFrame(self.X_train).rename(columns=str)

        # 2. Ensure y_train is a Series, using X_train's index for alignment.
        if not isinstance(self.y_train, (pd.Series, pd.DataFrame)):
            self.y_train = pd.Series(self.y_train, index=self.X_train.index)

        # --- CRITICAL FIX for H2O Stacked Ensemble response column mismatch ---
        # Enforce a consistent name for the target variable series. This prevents
        # the "response_column must match" error in H2O StackedEnsemble.
        self.y_train.name = "outcome"

        # --- OPTIMIZATION: Drop ID column if present ---
        # client_idcode is metadata and should not be used for training.
        # It can cause issues with models like CatBoost (high cardinality string) and slow down training.
        if "client_idcode" in self.X_train.columns:
            self.logger.debug("Dropping 'client_idcode' from training data.")
            self.X_train = self.X_train.drop(columns=["client_idcode"], errors="ignore")
            if isinstance(self.X_test, pd.DataFrame):
                self.X_test = self.X_test.drop(
                    columns=["client_idcode"], errors="ignore"
                )
            if isinstance(self.X_test_orig, pd.DataFrame):
                self.X_test_orig = self.X_test_orig.drop(
                    columns=["client_idcode"], errors="ignore"
                )

        max_param_space_iter_value = (
            self.global_params.max_param_space_iter_value
        )  # hard limit on param space exploration

        # Allow local override for max_param_space_iter_value
        if (
            self.ml_grid_object_iter.local_param_dict.get("max_param_space_iter_value")
            is not None
        ):
            max_param_space_iter_value = self.ml_grid_object_iter.local_param_dict.get(
                "max_param_space_iter_value"
            )

        if "svc" in method_name.lower():
            self.logger.info(
                "Applying StandardScaler for SVC to prevent convergence issues."
            )
            scaler = StandardScaler()
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

        # --- PERFORMANCE FIX for testing ---
        # Use a much faster CV strategy when in test_mode.
        # This MUST be defined before HyperparameterSearch is instantiated.
        if getattr(self.global_parameters, "test_mode", False):
            self.logger.info("Test mode enabled. Using fast KFold(n_splits=2) for CV.")
            self.cv = KFold(n_splits=2, shuffle=True, random_state=1)
        else:
            # Use the full, robust CV strategy for production runs
            self.cv = RepeatedKFold(
                # Using 2 splits for faster iteration and larger training folds.
                n_splits=2,
                n_repeats=2,
                random_state=1,
            )

        start = time.time()

        current_algorithm = algorithm_implementation

        # Silence verbose models like CatBoost to keep logs clean
        if "catboost" in method_name.lower() and hasattr(
            current_algorithm, "set_params"
        ):
            ml_grid_object.logger.info(
                "Silencing CatBoost verbose output and file writing."
            )
            current_algorithm.set_params(verbose=0, allow_writing_files=False)

        # Check for GPU availability and set device for torch-based models
        if "simbsig" in str(type(algorithm_implementation)):
            if not torch.cuda.is_available():
                self.logger.info(
                    "No CUDA GPU detected. Forcing simbsig model to use CPU."
                )
                if hasattr(current_algorithm, "set_params"):
                    current_algorithm.set_params(device="cpu")
            else:
                self.logger.info(
                    "CUDA GPU detected. Allowing simbsig model to use GPU."
                )

        self.logger.debug(f"Algorithm implementation: {algorithm_implementation}")

        parameters = parameter_space  # Keep a reference to the original

        if ml_grid_object.verbose >= 3:
            self.logger.debug(
                f"algorithm_implementation: {algorithm_implementation}, type: {type(algorithm_implementation)}"
            )

        # Validate parameters for non-Bayesian searches
        if not self.global_params.bayessearch:
            parameters = validate_parameters_helper(
                algorithm_implementation=algorithm_implementation,
                parameters=parameter_space,
                ml_grid_object=ml_grid_object,
            )

        # --- FIX for skopt ValueError ---
        # If using Bayesian search, ensure all list-based parameters are wrapped
        # in skopt.space.Categorical to prevent "can only convert an array of size 1" error.
        if self.global_params.bayessearch:
            self.logger.debug("Validating parameter space for Bayesian search...")
            if isinstance(
                parameter_space, list
            ):  # For models like LogisticRegression with multiple dicts
                # This part remains the same as it handles lists of dictionaries correctly.
                for i, space in enumerate(parameter_space):
                    new_space = {}
                    for key, value in space.items():
                        # --- REFINED FIX for skopt ValueError ---
                        # Check if the value is a list of potential choices that needs wrapping.
                        # This is true if it's a list/array, not already a skopt space,
                        # and its elements are not lists themselves (e.g., for H2O's 'hidden' param).
                        is_list_of_choices = (
                            isinstance(value, (list, np.ndarray))
                            and value
                            and not isinstance(value[0], list)
                        )
                        if is_list_of_choices and not is_skopt_space(value):
                            self.logger.warning(
                                f"Auto-correcting param '{key}' for BayesSearch: wrapping list in Categorical."
                            )
                            new_space[key] = Categorical(value)
                        else:  # It's a skopt object, a single value, or a list of lists (like for 'hidden')
                            new_space[key] = value
                    parameter_space[i] = new_space
            elif isinstance(parameter_space, dict):  # For standard single-dict spaces
                # This is the key change: iterate and build a new dictionary
                # to avoid issues with modifying a dictionary while iterating.
                new_parameter_space = {}
                for key, value in parameter_space.items():
                    # --- REFINED FIX for skopt ValueError ---
                    is_list_of_choices = (
                        isinstance(value, (list, np.ndarray))
                        and value
                        and not isinstance(value[0], list)
                    )
                    if is_list_of_choices and not is_skopt_space(value):
                        self.logger.warning(
                            f"Auto-correcting param '{key}' for BayesSearch: wrapping list in Categorical."
                        )
                        new_parameter_space[key] = Categorical(value)
                    else:  # It's a skopt object, a single value, or a list of lists (like for 'hidden')
                        new_parameter_space[key] = value
                parameter_space = new_parameter_space
            # Ensure the 'parameters' variable points to the updated parameter_space for Bayes search
            parameters = parameter_space

        # Use the new n_iter parameter from the config
        # Default to 2 if not present, preventing AttributeError
        try:
            n_iter_v = getattr(self.global_params, "n_iter", 2)
            if n_iter_v is None:
                n_iter_v = 2
            n_iter_v = int(n_iter_v)
        except (ValueError, TypeError):
            self.logger.warning(
                "Invalid or missing n_iter in global_params. Defaulting to 2."
            )
            n_iter_v = 2

        # Allow local override from run_params/local_param_dict
        local_n_iter = self.ml_grid_object_iter.local_param_dict.get("n_iter")
        if local_n_iter is not None:
            try:
                n_iter_v = int(local_n_iter)
                self.logger.info(
                    f"Overriding global n_iter with local value: {n_iter_v}"
                )
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid local n_iter value: {local_n_iter}. Ignoring override."
                )

        if max_param_space_iter_value is not None:
            if n_iter_v > max_param_space_iter_value:
                self.logger.info(
                    f"Capping n_iter ({n_iter_v}) to max_param_space_iter_value ({max_param_space_iter_value})"
                )
                n_iter_v = max_param_space_iter_value

        # For GridSearchCV, n_iter is not used, but we calculate the grid size for logging.
        if not self.global_params.bayessearch and not random_grid_search:
            pg = len(ParameterGrid(parameter_space))
            self.logger.info(f"Parameter grid size: {pg}")
        else:
            # For Random and Bayes search, log the number of iterations
            self.logger.info(f"Using n_iter={n_iter_v} for search.")

        # Calculate pg for logging purposes
        pg = (
            len(ParameterGrid(parameter_space))
            if not self.global_params.bayessearch
            else "N/A"
        )

        # Dynamically adjust KNN parameter space for small datasets
        if "kneighbors" in method_name.lower() or "simbsig" in method_name.lower():
            self._adjust_knn_parameters(parameter_space)
            self.logger.debug(
                "Adjusted KNN n_neighbors parameter space to prevent errors on small CV folds."
            )

        # Check if dataset is too small for CatBoost
        if "catboost" in method_name.lower():
            min_samples_required = 10  # CatBoost needs a reasonable amount of data
            if len(self.X_train) < min_samples_required:
                self.logger.warning(
                    f"Dataset too small for CatBoost ({len(self.X_train)} samples < {min_samples_required} required). "
                    f"Skipping {method_name}."
                )
                # Return early with default scores
                self.grid_search_cross_validate_score_result = 0.5
                return

        # Dynamically adjust CatBoost subsample parameter for small datasets
        if "catboost" in method_name.lower():
            self._adjust_catboost_parameters(parameter_space)
            self.logger.debug(
                "Adjusted CatBoost subsample parameter space to prevent errors on small CV folds."
            )

        # --- CRITICAL FIX for H2OStackedEnsemble ---
        # The special handling logic has been moved inside the H2OStackedEnsembleClassifier
        # class itself, making it a self-contained scikit-learn meta-estimator.
        # No special orchestration is needed here anymore.

        # --- OPTIMIZATION: Force Sequential Search for H2O/GPU Models ---
        # Save original n_jobs to restore later. This prevents HyperparameterSearch
        # from spawning parallel jobs for models that are not thread/process safe
        # or have their own internal parallelism (like H2O).
        original_grid_n_jobs = self.global_parameters.grid_n_jobs
        if is_gpu_model or is_h2o_model:
            self.global_parameters.grid_n_jobs = 1

        try:
            # Instantiate and run the hyperparameter grid/random search
            search = HyperparameterSearch(
                algorithm=current_algorithm,
                parameter_space=parameters,  # Use the validated/modified parameters
                method_name=method_name,
                global_params=self.global_parameters,
                sub_sample_pct=self.sub_sample_param_space_pct,  # Explore 50% of the parameter space
                max_iter=n_iter_v,  # Maximum iterations for randomized search
                ml_grid_object=ml_grid_object,
                cv=self.cv,
            )

            if self.global_parameters.verbose >= 3:
                self.logger.debug("Running hyperparameter search")

            # Define default scores early to handle timeouts in search phase
            default_scores = {
                "test_accuracy": np.array([0.5]),
                "test_f1": np.array([0.5]),
                "test_auc": np.array([0.5]),
                "fit_time": np.array([0]),
                "score_time": np.array([0]),
                "train_score": np.array([0.5]),
                "test_recall": np.array([0.5]),
            }

            failed = False
            scores = None

            # Initialize start_time early
            start_time = time.time()

            try:
                # Verify initial index alignment
                try:
                    assert_index_equal(self.X_train.index, self.y_train.index)
                    ml_grid_object.logger.debug(
                        "Index alignment PASSED before search.run_search"
                    )
                except AssertionError:
                    ml_grid_object.logger.error(
                        "Index alignment FAILED before search.run_search"
                    )
                    raise

                # Ensure y_train is a Series for consistency
                if not isinstance(self.y_train, pd.Series):
                    ml_grid_object.logger.error(
                        f"y_train is not a pandas Series, but {type(self.y_train)}. Converting to Series."
                    )
                    self.y_train = pd.Series(self.y_train, index=self.X_train.index)

                # CRITICAL FIX: Reset indices to ensure integer-based indexing for sklearn
                # This prevents "String indexing is not supported with 'axis=0'" errors
                X_train_reset = self.X_train.reset_index(drop=True)
                y_train_reset = self.y_train.reset_index(drop=True)

                ml_grid_object.logger.debug(
                    f"X_train index after reset: {X_train_reset.index[:5]}"
                )
                ml_grid_object.logger.debug(
                    f"y_train index after reset: {y_train_reset.index[:5]}"
                )

                # --- OPTIMIZATION: Convert y to numpy for ALL models ---
                # This avoids expensive sklearn type_of_target checks on Pandas Series (overhead seen in profiling)
                # Most sklearn models handle numpy arrays efficiently.
                y_train_search = self._optimize_y(y_train_reset)

                # --- OPTIMIZATION: Skip parameter validation overhead ---
                # Use set_config to ensure it propagates to all internal calls
                with sklearn.config_context(skip_parameter_validation=True):
                    # Pass reset data to search
                    if is_h2o_model:
                        try:
                            import h2o

                            h2o.no_progress()
                        except Exception:
                            pass

                    # --- OPTIMIZATION: Force threading backend for search ---
                    # Prevents 'loky' overhead (abort_everything ~273s) which occurs even with n_jobs=1
                    with joblib.parallel_backend("threading"):
                        current_algorithm = search.run_search(
                            X_train_reset, y_train_search
                        )

            except TimeoutError:
                self.logger.warning("Timeout occurred during hyperparameter search.")
                failed = "Timeout"
                scores = default_scores

            except KeyboardInterrupt:
                if "catboost" in method_name.lower():
                    self.logger.warning(
                        "KeyboardInterrupt detected during hyperparameter search. "
                        "This is likely a signal handling artifact (e.g. from CatBoost) triggered by the timeout. "
                        "Treating as a timeout."
                    )
                    failed = "KeyboardInterrupt"
                    scores = default_scores
                else:
                    raise

            except Exception as e:
                if "dual coefficients or intercepts are not finite" in str(e):
                    self.logger.warning(
                        f"SVC failed to fit due to data issues: {e}. Returning default score."
                    )
                    self.grid_search_cross_validate_score_result = 0.5
                    return

                # Log the error and re-raise it to stop the entire execution,
                # allowing the main loop in main.py to handle it based on error_raise.
                self.logger.error(
                    f"An exception occurred during hyperparameter search for {method_name}: {e}",
                    exc_info=True,
                )
                raise e
        finally:
            # Restore the original grid_n_jobs setting
            self.global_parameters.grid_n_jobs = original_grid_n_jobs

        # --- PERFORMANCE FIX for testing ---
        # If in test_mode, we have already verified that the search runs without crashing.
        # We can skip the final, slow cross-validation and return a dummy score.
        if not failed and getattr(self.global_parameters, "test_mode", False):
            self.logger.info(
                "Test mode enabled. Skipping final cross-validation for speed."
            )
            self.grid_search_cross_validate_score_result = 0.5  # Return a valid float
            # Final cleanup for H2O models
            self._shutdown_h2o_if_needed(current_algorithm)
            return

        if not failed and self.global_parameters.verbose >= 3:
            self.logger.debug("Fitting final model")

        # In production, we re-fit the best estimator on the full training data before CV.
        # In test_mode, the estimator from the search is already fitted, and re-fitting
        # can invalidate complex models like H2OStackedEnsemble before the final assert.

        metric_list = self.metric_list

        # Catch only one class present AUC not defined (check only if not already failed)
        # Optimization: Use pandas nunique() which is faster than converting to numpy and sorting
        if not failed and self.y_train.nunique() < 2:
            raise ValueError(
                "Only one class present in y_train. ROC AUC score is not defined "
                "in that case. grid_search_cross_validate>>>cross_validate"
            )

        if not failed and self.global_parameters.verbose >= 1:
            self.logger.info("Getting cross validation scores")
            self.logger.debug(
                f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}"
            )
            self.logger.debug(f"y_train value counts:\n{self.y_train.value_counts()}")

        # Set a time threshold in seconds
        time_threshold = 60  # For example, 60 seconds

        # --- CRITICAL FIX for H2O multiprocessing error ---
        # H2O models cannot be pickled and sent to other processes for parallel
        # execution with joblib. We must detect if the current algorithm is an
        # H2O model and, if so, force n_jobs=1 for cross_validate.
        # (H2O_MODEL_TYPES is now defined at module level)

        # Keras/TensorFlow models also require single-threaded execution.
        keras_model_types = (NeuralNetworkClassifier, KerasClassifierClass)

        is_h2o_model = isinstance(current_algorithm, H2O_MODEL_TYPES)
        is_keras_model = isinstance(current_algorithm, keras_model_types)

        # H2O and Keras models require single-threaded execution for CV
        final_cv_n_jobs = 1 if is_h2o_model or is_keras_model else grid_n_jobs
        if final_cv_n_jobs == 1:
            self.logger.debug(
                "H2O or Keras model detected. Forcing n_jobs=1 for final cross-validation."
            )

        try:
            if failed:
                raise TimeoutError

            # H2O models require pandas DataFrames with column names, while other
            # sklearn models can benefit from using NumPy arrays.
            if isinstance(current_algorithm, H2O_MODEL_TYPES):
                X_train_final = self.X_train  # Pass DataFrame directly
                # Optimization: Pass numpy array for y to avoid pandas overhead in sklearn checks
                if isinstance(self.y_train.dtype, pd.CategoricalDtype):
                    y_train_final = self.y_train.cat.codes.values
                else:
                    y_train_final = self.y_train.values
            else:
                X_train_final = self.X_train.values  # Use NumPy array for other models
                # Optimization: Pass numpy array for y to avoid pandas overhead in sklearn
                y_train_final = self._optimize_y(self.y_train)

            scores = None

            # Check for user override to force second CV
            # Check local params first, then global params
            force_second_cv = self.ml_grid_object_iter.local_param_dict.get(
                "force_second_cv", getattr(self.global_params, "force_second_cv", False)
            )

            if force_second_cv:
                self.logger.info(
                    "force_second_cv is True. Skipping cached result extraction to run fresh cross-validation."
                )

            # Check if we can reuse results from HyperparameterSearch
            if (
                not force_second_cv
                and hasattr(current_algorithm, "cv_results_")
                and hasattr(current_algorithm, "best_index_")
            ):
                try:
                    self.logger.info(
                        "Using cached cross-validation results from HyperparameterSearch."
                    )
                    results = current_algorithm.cv_results_
                    index = current_algorithm.best_index_
                    n_splits = self.cv.get_n_splits()

                    temp_scores = {}
                    # Extract fit and score times
                    if "split0_fit_time" in results:
                        temp_scores["fit_time"] = np.array(
                            [
                                results[f"split{k}_fit_time"][index]
                                for k in range(n_splits)
                            ]
                        )
                    else:
                        # Fallback: Use mean time repeated if split times are missing (e.g. BayesSearchCV)
                        temp_scores["fit_time"] = np.full(
                            n_splits, results["mean_fit_time"][index]
                        )

                    if "split0_score_time" in results:
                        temp_scores["score_time"] = np.array(
                            [
                                results[f"split{k}_score_time"][index]
                                for k in range(n_splits)
                            ]
                        )
                    else:
                        # Fallback: Use mean score time.
                        # We use .get() with a default that is safe to index into if the key is missing.
                        # If 'mean_score_time' is missing, we default to a list of 0s large enough to cover 'index'.
                        default_times = np.zeros(index + 1)
                        temp_scores["score_time"] = np.full(
                            n_splits,
                            results.get("mean_score_time", default_times)[index],
                        )

                    # Extract metric scores
                    for metric in self.metric_list:
                        # Test scores
                        test_key = f"test_{metric}"
                        temp_scores[test_key] = np.array(
                            [
                                results[f"split{k}_test_{metric}"][index]
                                for k in range(n_splits)
                            ]
                        )
                        # Train scores (if available)
                        train_key = f"train_{metric}"
                        train_col = (
                            f"split0_train_{metric}"  # Check existence on first split
                        )
                        if train_col in results:
                            temp_scores[train_key] = np.array(
                                [
                                    results[f"split{k}_train_{metric}"][index]
                                    for k in range(n_splits)
                                ]
                            )
                    scores = temp_scores
                except Exception as e:
                    self.logger.warning(
                        f"Could not extract cached CV results: {e}. Falling back to standard CV."
                    )
                    scores = None

            if scores is None:
                if isinstance(
                    current_algorithm, (KerasClassifier, KerasClassifierClass)
                ):
                    self.logger.debug("Fitting Keras model with internal CV handling.")
                    y_train_values = self.y_train.values
                    current_algorithm.fit(self.X_train, y_train_values, cv=self.cv)
                    # Since fit already did the CV, create a dummy scores dictionary.
                    scores = {
                        "test_roc_auc": [
                            current_algorithm.score(self.X_test, self.y_test.values)
                        ]
                    }
                else:
                    # For all other models, perform standard cross-validation.
                    # Note: current_algorithm is already fitted on full X_train by HyperparameterSearch (refit=True)
                    # so we do not need to call .fit() again here.

                    # --- OPTIMIZATION: Skip parameter validation overhead (99s) ---
                    with sklearn.config_context(skip_parameter_validation=True):
                        # Ensure H2O progress is disabled before CV
                        if is_h2o_model:
                            try:
                                import h2o

                                h2o.no_progress()
                            except Exception:
                                pass

                        # --- OPTIMIZATION: Always use threading backend ---
                        # The overhead of 'loky' (multiprocessing) is excessive (sleep + memmapping ~300s).
                        # Threading is sufficient and much faster for this pipeline.
                        backend = "threading"

                        with joblib.parallel_backend(backend):
                            scores = cross_validate(
                                current_algorithm,
                                X_train_final,
                                y_train_final,  # Use optimized y (numpy for sklearn, Series for H2O)
                                scoring=self.metric_list,
                                cv=self.cv,
                                n_jobs=final_cv_n_jobs,  # Use adjusted n_jobs
                                pre_dispatch="2*n_jobs",
                                error_score=self.error_raise,  # Raise error if cross-validation fails
                            )

                    # Pre-compile the predict function for Keras/TF models to avoid retracing warnings.
                    # This is done AFTER fitting and before cross-validation.
                    if isinstance(
                        current_algorithm,
                        (
                            KerasClassifier,
                            KerasClassifierClass,
                            NeuralNetworkClassifier,
                        ),
                    ):
                        try:
                            self.logger.debug(
                                "Pre-compiling TensorFlow predict function to avoid retracing."
                            )
                            n_features = self.X_train.shape[1]
                            # Define an input signature that allows for variable batch size.
                            input_signature = [
                                tf.TensorSpec(
                                    shape=(None, n_features), dtype=tf.float32
                                )
                            ]
                            # Access the underlying Keras model via .model_
                            current_algorithm.model_.predict.get_concrete_function(
                                input_signature
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Could not pre-compile TF function. Performance may be impacted. Error: {e}"
                            )

        except XGBoostError as e:
            if "cuda" in str(e).lower() or "memory" in str(e).lower():
                self.logger.warning(
                    "GPU memory error detected during cross-validation, falling back to CPU..."
                )
                current_algorithm.set_params(tree_method="hist")

                try:
                    scores = cross_validate(
                        current_algorithm,
                        X_train_final,
                        y_train_final,  # Use optimized y
                        scoring=self.metric_list,
                        cv=self.cv,
                        n_jobs=final_cv_n_jobs,  # Use adjusted n_jobs
                        pre_dispatch="2*n_jobs",
                        error_score=self.error_raise,  # Raise error if cross-validation fails
                    )
                except Exception as e:
                    self.logger.error(
                        f"An unexpected error occurred during cross-validation attempt 2: {e}",
                        exc_info=True,
                    )
                    self.logger.warning("Returning default scores")
                    failed = True
                    scores = default_scores  # Use default scores for other errors

        except ValueError as e:
            # Handle specific ValueError if AdaBoostClassifier fails due to poor performance
            if (
                "BaseClassifier in AdaBoostClassifier ensemble is worse than random"
                in str(e)
            ):
                self.logger.warning(f"AdaBoostClassifier failed: {e}")
                self.logger.warning(
                    "Skipping AdaBoostClassifier due to poor base classifier performance."
                )

                # Set default scores if the AdaBoostClassifier fails
                failed = True
                scores = default_scores  # Use default scores

            else:
                self.logger.error(
                    f"An unexpected ValueError occurred during cross-validation: {e}",
                    exc_info=True,
                )
                failed = True
                scores = default_scores  # Use default scores for other errors

        except RuntimeError as e:
            # raise e  # raise h2o errors to aid development
            # --- FIX for UnboundLocalError with H2OStackedEnsemble ---
            # Catch any RuntimeError, which can be raised by H2O models during fit
            # (e.g., base model training failure) or predict.
            self.logger.error(
                f"A RuntimeError occurred during cross-validation (often H2O related): {e}",
                exc_info=True,
            )
            self.logger.warning("Returning default scores.")
            failed = True
            scores = default_scores

        except TimeoutError:
            self.logger.warning("Timeout occurred during cross-validation.")
            failed = "Timeout"
            scores = default_scores

        except KeyboardInterrupt:
            if "catboost" in method_name.lower():
                self.logger.warning(
                    "KeyboardInterrupt detected during cross-validation. "
                    "This is likely a signal handling artifact (e.g. from CatBoost) triggered by the timeout. "
                    "Treating as a timeout."
                )
                failed = "KeyboardInterrupt"
                scores = default_scores
            else:
                raise

        except Exception as e:
            # Catch any other general exceptions and log them
            self.logger.error(
                f"An unexpected error occurred during cross-validation: {e}",
                exc_info=True,
            )
            failed = True
            scores = default_scores  # Use default scores if an error occurs

        # End the timer
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        if self.global_parameters.verbose >= 1:
            # Print a warning if the execution time exceeds the threshold
            if elapsed_time > time_threshold:
                self.logger.warning(
                    f"Cross-validation took too long ({elapsed_time:.2f} seconds). "
                    "Consider optimizing the parameters or reducing CV folds."
                )
            else:
                self.logger.info(
                    f"Cross-validation for {method_name} completed in {elapsed_time:.2f} seconds."
                )

        current_algorithm_scores = scores
        #     scores_tuple_list.append((method_name, current_algorithm_scores, grid))

        if self.global_parameters.verbose >= 4:

            debug_print_statements_class(scores).debug_print_scores()

        plot_auc = False
        if plot_auc:
            # This was passing a classifier trained on the test dataset....
            self.logger.debug("Plotting AUC is disabled.")

            # plot_auc_results(current_algorithm, self.X_test_orig[self.X_train.columns], self.y_test_orig, self.cv)
            # plot_auc_results(grid.best_estimator_, X_test_orig, self.y_test_orig, cv)

        #         this should be x_test...?
        try:
            best_pred_orig = current_algorithm.predict(self.X_test)  # exp
        except Exception:
            best_pred_orig = np.zeros(len(self.X_test))

        # Call the update_score_log method on the provided instance
        if self.project_score_save_class_instance:
            self.project_score_save_class_instance.update_score_log(
                ml_grid_object=ml_grid_object,
                scores=scores,
                best_pred_orig=best_pred_orig,
                current_algorithm=current_algorithm,
                method_name=method_name,
                pg=pg,
                start=start,
                n_iter_v=n_iter_v,
                failed=failed,
            )
        else:
            self.logger.warning(
                "No project_score_save_class_instance provided. Skipping score logging."
            )

        # calculate metric for optimisation
        try:
            y_test_np = (
                self.y_test.values if hasattr(self.y_test, "values") else self.y_test
            )
            auc = metrics.roc_auc_score(y_test_np, best_pred_orig)
        except Exception:
            auc = 0.5

        self.grid_search_cross_validate_score_result = auc

        self._shutdown_h2o_if_needed(current_algorithm)

    def _optimize_y(self, y):
        """Helper to optimize y for sklearn/H2O to reduce type_of_target overhead."""
        # Convert to numpy if it's a Series or Categorical
        if hasattr(y, "dtype") and isinstance(y.dtype, pd.CategoricalDtype):
            y_opt = y.cat.codes.values
        elif hasattr(y, "values"):
            y_opt = y.values
        else:
            y_opt = y

        # Force integer encoding
        if not pd.api.types.is_integer_dtype(y_opt):
            try:
                y_opt = y_opt.astype(int)
            except (ValueError, TypeError):
                y_opt, _ = pd.factorize(y_opt, sort=True)
                y_opt = y_opt.astype(int)

        # Ensure contiguous array for speed in np.unique and other ops
        return np.ascontiguousarray(y_opt)

    def _adjust_knn_parameters(self, parameter_space: Union[Dict, List[Dict]]):
        """
        Dynamically adjusts the 'n_neighbors' parameter for KNN-based models
        to prevent errors on small datasets during cross-validation.
        """
        n_splits = self.cv.get_n_splits()

        # --- CRITICAL FIX: Correctly calculate the training fold size ---
        # The previous calculation was incorrect for some CV strategies.
        # This method is robust: create a dummy split to get the exact train fold size.
        dummy_indices = np.arange(len(self.X_train))
        train_indices, _ = next(self.cv.split(dummy_indices))
        n_samples_train_fold = len(train_indices)
        n_samples_test_fold = len(self.X_train) - n_samples_train_fold
        max_n_neighbors = max(1, n_samples_train_fold)

        self.logger.debug(
            f"KNN constraints - train_fold_size={n_samples_train_fold}, "
            f"test_fold_size={n_samples_test_fold}, max_n_neighbors={max_n_neighbors}"
        )

        def adjust_param(param_value):
            if is_skopt_space(param_value):
                # For skopt.space objects, adjust the upper bound
                new_high = min(param_value.high, max_n_neighbors)
                new_low = min(param_value.low, new_high)
                param_value.high = new_high
                param_value.low = new_low
                self.logger.debug(
                    f"Adjusted skopt space: low={new_low}, high={new_high}"
                )
            elif isinstance(param_value, (list, np.ndarray)):
                # For lists, filter the values
                new_param_value = [n for n in param_value if n <= max_n_neighbors]
                if not new_param_value:
                    self.logger.warning(
                        f"All n_neighbors values filtered out. Using [{max_n_neighbors}]"
                    )
                    return [max_n_neighbors]
                self.logger.debug(f"Filtered n_neighbors list: {new_param_value}")
                return new_param_value
            return param_value

        if isinstance(parameter_space, list):
            for params in parameter_space:
                if "n_neighbors" in params:
                    params["n_neighbors"] = adjust_param(params["n_neighbors"])
        elif isinstance(parameter_space, dict) and "n_neighbors" in parameter_space:
            parameter_space["n_neighbors"] = adjust_param(
                parameter_space["n_neighbors"]
            )

    def _adjust_catboost_parameters(self, parameter_space: Union[Dict, List[Dict]]):
        """
        Dynamically adjusts the 'subsample' parameter for CatBoost to prevent
        errors on small datasets during cross-validation.
        """
        n_splits = self.cv.get_n_splits()
        # Correctly calculate the size of the smallest training fold.
        n_samples_in_fold = len(self.X_train) - (len(self.X_train) // n_splits)

        # Ensure n_samples_in_fold is at least 1 to avoid division by zero
        n_samples_in_fold = max(1, n_samples_in_fold)

        # If the training fold is extremely small, force subsample to 1.0
        # to prevent CatBoost from failing on constant features.
        if n_samples_in_fold <= 2:
            min_subsample = 1.0
        else:
            # The minimum subsample value must be > 1/n_samples to ensure at least one sample is chosen
            min_subsample = 1.0 / n_samples_in_fold

        def adjust_param(param_value):
            if is_skopt_space(param_value):
                # For skopt.space objects (Real), adjust the lower bound
                new_low = max(param_value.low, min_subsample)
                # Ensure the new low is not higher than the high
                if new_low > param_value.high:
                    new_low = param_value.high
                param_value.low = new_low
                # If the fold is tiny, force the entire space to be 1.0
                if n_samples_in_fold <= 2:
                    param_value.low = param_value.high = 1.0
            elif isinstance(param_value, (list, np.ndarray)):
                # For lists, filter the values
                new_param_value = [s for s in param_value if s >= min_subsample]
                if not new_param_value:
                    # If all values are filtered out, use the smallest valid value
                    return [
                        (
                            min(p for p in param_value if p > 0)
                            if any(p > 0 for p in param_value)
                            else 1.0
                        )
                    ]
                return new_param_value
            # If the fold is tiny, force subsample to 1.0
            if n_samples_in_fold <= 2:
                return [1.0] if isinstance(param_value, list) else 1.0
            return param_value

        if isinstance(parameter_space, list):
            for params in parameter_space:
                if "subsample" in params:
                    params["subsample"] = adjust_param(params["subsample"])
        elif isinstance(parameter_space, dict) and "subsample" in parameter_space:
            parameter_space["subsample"] = adjust_param(parameter_space["subsample"])

        # Also adjust 'rsm' (colsample_bylevel) which can cause the same issue
        if isinstance(parameter_space, list):
            for params in parameter_space:
                if "rsm" in params:
                    params["rsm"] = adjust_param(params["rsm"])
        elif isinstance(parameter_space, dict) and "rsm" in parameter_space:
            parameter_space["rsm"] = adjust_param(parameter_space["rsm"])

    def _shutdown_h2o_if_needed(self, algorithm: Any):
        """Safely shuts down the H2O cluster if the algorithm is an H2O model."""
        # Use the module-level tuple
        if isinstance(algorithm, H2O_MODEL_TYPES):
            # --- FIX for repeated H2O cluster shutdown ---
            # We no longer shut down the cluster after each model.
            # The cluster is now managed globally and should be shut down
            # at the end of the entire experiment run.
            import h2o

            cluster = h2o.cluster()
            if cluster and cluster.is_running():
                self.logger.info(
                    "H2O model finished. Leaving cluster running for next H2O model."
                )
            # The shutdown call was removed from H2OBaseClassifier. The cluster is managed globally.


def dummy_auc() -> float:
    """Returns a constant AUC score of 0.5.

    This function is intended as a placeholder or for use in scenarios where
    a valid AUC score cannot be calculated but a value is required.

    Returns:
        float: A constant value of 0.5.
    """
    return 0.5


# Create a scorer using make_scorer
# dummy_auc_scorer = make_scorer(dummy_auc)


def scale_data(X_train: pd.DataFrame) -> pd.DataFrame:
    """Scales the data to a [0, 1] range if it's not already scaled.

    Args:
        X_train (pd.DataFrame): Training features.

    Returns:
        pd.DataFrame: Scaled training features.
    """
    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Check if data is already scaled
    min_val = X_train.min().min()
    max_val = X_train.max().max()

    # If data is not scaled, then scale it
    if min_val < 0 or max_val > 1:
        # Fit and transform the data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        return X_train_scaled
    else:
        # If data is already scaled, return it as is
        return X_train
