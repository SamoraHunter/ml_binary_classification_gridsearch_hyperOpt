import time
import traceback
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from numpy import absolute, mean, std
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from IPython.display import display
from pandas.testing import assert_index_equal
from xgboost.core import XGBoostError

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import *
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_validate,
)

from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.pipeline.hyperparameter_search import HyperparameterSearch
from ml_grid.util.debug_print_statements import debug_print_statements_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class
from ml_grid.util.validate_parameters import validate_parameters_helper
from sklearn.preprocessing import MinMaxScaler
from ml_grid.util.bayes_utils import calculate_combinations, is_skopt_space
from skopt.space import Categorical

class grid_search_crossvalidate:

    def __init__(
        self,
        algorithm_implementation: Any,
        parameter_space: Union[Dict, List[Dict]],
        method_name: str,
        ml_grid_object: Any,
        sub_sample_parameter_val: int = 100,
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
        """
        # Set each warning filter individually for robustness
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.logger = logging.getLogger('ml_grid')

        self.global_params = global_parameters

        self.verbose = self.global_params.verbose

        if self.verbose < 8:
            self.logger.debug("Clearing output.")
            clear_output(wait=True)

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        random_grid_search = self.global_params.random_grid_search

        self.sub_sample_parameter_val = sub_sample_parameter_val

        grid_n_jobs = self.global_params.grid_n_jobs

        # Configure GPU usage and job limits for specific models
        if "keras" in method_name.lower() or "xgb" in method_name.lower() or "catboost" in method_name.lower():
            grid_n_jobs = 1
            try:
                gpu_devices = tf.config.experimental.list_physical_devices("GPU")
                for device in gpu_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except Exception as e:
                self.logger.warning(f"Could not configure GPU for TensorFlow: {e}")

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

        max_param_space_iter_value = self.global_params.max_param_space_iter_value # hard limit on param space exploration

        if "svc" in method_name.lower():
            self.X_train = scale_data(self.X_train)
            self.X_test = scale_data(self.X_test)

        # CRITICAL: Initialize the cross-validation object before it is used.
        self.cv = RepeatedKFold(
            n_splits=max(2, min(len(self.X_train), 2) + 1), 
            n_repeats=2, 
            random_state=1
        )

        start = time.time()

        current_algorithm = algorithm_implementation

        # Silence verbose models like CatBoost to keep logs clean
        if "catboost" in method_name.lower() and hasattr(current_algorithm, 'set_params'):
            ml_grid_object.logger.info("Silencing CatBoost verbose output.")
            current_algorithm.set_params(verbose=0)
        
        self.logger.info(f"Algorithm implementation: {algorithm_implementation}")

        parameters = parameter_space
        
        if(self.global_params.bayessearch is False):
            n_iter_v = np.nan
        else:
            n_iter_v = 2
        #     if(sub_sample_param_space):
        #         sub_sample_param_space_n = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space)))
        #         parameter_space random.sample(ParameterGrid(parameter_space), sub_sample_param_space_n)

        # Grid search over hyperparameter space, randomised.

        if ml_grid_object.verbose >= 3:
            self.logger.debug(f"algorithm_implementation: {algorithm_implementation}, type: {type(algorithm_implementation)}")
        
        if(self.global_params.bayessearch is False):
            # Validate parameters
            parameters = validate_parameters_helper(
                algorithm_implementation=algorithm_implementation,
                parameters=parameters,
                ml_grid_object=ml_grid_object,
            )

        # if random_grid_search:
        #     # n_iter_v = int(self.sub_sample_param_space_pct *  len(ParameterGrid(parameter_space))) + 2
        #     n_iter_v = int(len(ParameterGrid(parameter_space))) + 2

        #     if self.sub_sample_parameter_val < n_iter_v:
        #         n_iter_v = self.sub_sample_parameter_val
        #     if n_iter_v < 2:
        #         self.logger.warning("warn n_iter_v < 2")
        #         n_iter_v = 2
        #     if n_iter_v > max_param_space_iter_value:
        #         self.logger.warning(f"Warn n_iter_v > max_param_space_iter_value, setting {max_param_space_iter_value}")
        #         n_iter_v = max_param_space_iter_value

        #     grid = RandomizedSearchCV(
        #         current_algorithm,
        #         parameters,
        #         verbose=1,
        #         cv=[(slice(None), slice(None))],
        #         n_jobs=grid_n_jobs,
        #         n_iter=n_iter_v,
        #         # error_score=np.nan,
        #         error_score="raise",
        #     )
        # else:
        #     grid = GridSearchCV(
        #         current_algorithm,
        #         parameters,
        #         verbose=1,
        #         cv=[(slice(None), slice(None))],
        #         n_jobs=grid_n_jobs,
        #         error_score=np.nan,
        #     )  # Negate CV in param search for speed
        
        if not self.global_parameters.bayessearch:
            pg = ParameterGrid(parameter_space)
            pg = len(pg)
        else:
            pg = calculate_combinations(parameter_space, steps=n_iter_v) #untested n iter v
        #self.logger.debug(f"Approximate number of combinations: {approx_combinations}")
 
        if (random_grid_search and n_iter_v > 100000) or (
            random_grid_search == False and pg > 100000
        ):
            self.logger.warning(f"Grid too large. pg: {pg}, n_iter_v: {n_iter_v}")
            # raise Exception("grid too large", str(pg))

        if self.global_parameters.verbose >= 1:
            if random_grid_search:
                self.logger.info(
                    f"Randomized parameter grid size for {current_algorithm} \n : Full: {pg}, (mean * {self.sub_sample_param_space_pct}): {self.sub_sample_parameter_val}, current: {n_iter_v} "
                )

            else:
                self.logger.info(f"Parameter grid size: Full: {pg}")

        #grid.fit(self.X_train, self.y_train)
        if self.global_parameters.bayessearch:
            n_iter_v = pg + 2
        else:
            n_iter_v = int(len(ParameterGrid(parameter_space))) + 2 #review relevance and value

        if self.sub_sample_parameter_val < n_iter_v:
            n_iter_v = self.sub_sample_parameter_val
        if n_iter_v < 2:
            self.logger.warning("n_iter_v < 2, setting to 2")
            n_iter_v = 2
        if n_iter_v > max_param_space_iter_value:
            self.logger.warning(f"n_iter_v > max_param_space_iter_value, setting to {max_param_space_iter_value}.")
            n_iter_v = max_param_space_iter_value
        self.logger.info(f"n_iter_v = {n_iter_v}")

        # Dynamically adjust KNN parameter space for small datasets
        if "kneighbors" in method_name.lower():
            self._adjust_knn_parameters(parameter_space)

        # Instantiate and run the hyperparameter grid/random search
        search = HyperparameterSearch(
            algorithm=current_algorithm,
            parameter_space=parameter_space,
            method_name=method_name,
            global_params=self.global_parameters,
            sub_sample_pct=self.sub_sample_param_space_pct,  # Explore 50% of the parameter space
            max_iter=n_iter_v,         # Maximum iterations for randomized search
            ml_grid_object=ml_grid_object,
            cv=self.cv
        )

        if self.global_parameters.verbose >= 3:
            self.logger.debug("Running hyperparameter search")
        
        try:    
            # Verify initial index alignment
            try:
                assert_index_equal(self.X_train.index, self.y_train.index)
                ml_grid_object.logger.debug("Index alignment PASSED before search.run_search")
            except AssertionError:
                ml_grid_object.logger.error("Index alignment FAILED before search.run_search")
                raise

            # Ensure y_train is a Series for consistency
            if not isinstance(self.y_train, pd.Series):
                ml_grid_object.logger.error(f"y_train is not a pandas Series, but {type(self.y_train)}. Converting to Series.")
                self.y_train = pd.Series(self.y_train, index=self.X_train.index)

            # CRITICAL FIX: Reset indices to ensure integer-based indexing for sklearn
            # This prevents "String indexing is not supported with 'axis=0'" errors
            X_train_reset = self.X_train.reset_index(drop=True)
            y_train_reset = self.y_train.reset_index(drop=True)
            
            ml_grid_object.logger.debug(f"X_train index after reset: {X_train_reset.index[:5]}")
            ml_grid_object.logger.debug(f"y_train index after reset: {y_train_reset.index[:5]}")

            # Pass reset data to search
            current_algorithm = search.run_search(X_train_reset, y_train_reset)
            
        except XGBoostError as e:
            if 'cuda' in str(e).lower() or 'memory' in str(e).lower():
                self.logger.warning("GPU memory error detected, falling back to CPU...")
                 
                 # Change the tree_method in parameter_space dynamically
                if isinstance(parameter_space, list):
                    for param_dict in parameter_space:
                        if 'tree_method' in param_dict:
                            param_dict['tree_method'] = Categorical(['hist']) if self.global_params.bayessearch else ["hist"]
                elif isinstance(parameter_space, dict) and 'tree_method' in parameter_space:
                     parameter_space['tree_method'] = Categorical(['hist']) if self.global_params.bayessearch else ["hist"]
                 
                search = HyperparameterSearch(
                    algorithm=current_algorithm,
                    parameter_space=parameter_space,
                    method_name=method_name,
                    global_params=self.global_parameters,
                    sub_sample_pct=self.sub_sample_param_space_pct,
                    max_iter=n_iter_v,
                    ml_grid_object=ml_grid_object,
                    cv=self.cv
                )
                # Try again with CPU method and reset indices
                X_train_reset = self.X_train.reset_index(drop=True)
                y_train_reset = self.y_train.reset_index(drop=True)
                current_algorithm = search.run_search(X_train_reset, y_train_reset)
            else: 
                self.logger.error(f"Unknown XGBoostError: {e}", exc_info=True)
                raise
            
        except Exception as e:
            if "String indexing is not supported with 'axis=0'" in str(e):
                raise TypeError(
                    "Pandas indexing error: 'String indexing is not supported with 'axis=0''. "
                    "This typically happens when a pandas Series with a non-standard index is passed to a scikit-learn function. "
                    "Ensure that target variables (y_train) are converted to numpy arrays using `.values` before fitting or cross-validation."
                ) from e
            else:
                ml_grid_object.logger.error(f"Failed to run search in gridsearch cross validate: {e}", exc_info=True)
                # Re-raise the original exception to allow for higher-level handling if needed
                raise e


        if self.global_parameters.verbose >= 3:
            self.logger.debug("Fitting final model")
        #current_algorithm = grid.best_estimator_
        # Pass the DataFrame for the final fit to support models that need column names (e.g., LightGBM wrapper).
        # For cross-validation, we will use numpy arrays for performance and compatibility.
        y_train_values = self.y_train.values
        current_algorithm.fit(self.X_train, y_train_values)

        metric_list = self.metric_list

        # Catch only one class present AUC not defined:
        
        #dummy_auc_scorer = make_scorer(dummy_auc)
        if len(np.unique(self.y_train)) < 2:
            raise ValueError("Only one class present in y_train. ROC AUC score is not defined in that case. grid_search_cross_validate>>>cross_validate")

        if self.global_parameters.verbose >= 1:
            self.logger.info("Getting cross validation scores")
            self.logger.info(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            self.logger.info(f"y_train value counts:\n{self.y_train.value_counts()}")

        # Set a time threshold in seconds
        time_threshold = 60  # For example, 60 seconds

        start_time = time.time()

        # Define default scores (e.g., mean score of 0.5 for binary classification)
        # Default scores if cross-validation fails
        default_scores = {
            'test_accuracy': [0.5],   # Default to random classifier performance (0.5 for binary classification)
            'test_f1': [0.5],         # Default F1 score (again, 0.5 for random classification)
            'test_auc': [0.5],     # Default ROC AUC score (0.5 for random classifier) #is only auc not roc_auc?
            'fit_time': [0],           # No fitting time if the model fails
            'score_time': [0],         # No scoring time if the model fails
            'train_score': [0.5],      # Default train score
            'test_recall':[0.5]
            #'test_auc': [0.5] # ?
        }
        
        failed = False

        try:
            # Perform the cross-validation
            X_train_final_np = self.X_train.values
            scores = cross_validate(
                current_algorithm,
                X_train_final_np,
                y_train_values, # This is already a numpy array
                scoring=self.metric_list,
                cv=self.cv,
                n_jobs=grid_n_jobs,  # Full CV on final best model
                pre_dispatch=80,
                error_score=self.error_raise,  # Raise error if cross-validation fails
            )
            
            
        except XGBoostError as e:
            if 'cuda' in str(e).lower() or 'memory' in str(e).lower():
                self.logger.warning("GPU memory error detected during cross-validation, falling back to CPU...")
                current_algorithm.set_params(tree_method='hist') 
                
                try:
                    X_train_final_np = self.X_train.values
                    scores = cross_validate(
                        current_algorithm,
                        X_train_final_np,
                        y_train_values,
                        scoring=self.metric_list,
                        cv=self.cv,
                        n_jobs=grid_n_jobs,  # Full CV on final best model
                        pre_dispatch=80,
                        error_score=self.error_raise,  # Raise error if cross-validation fails
                    )
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred during cross-validation attempt 2: {e}", exc_info=True)
                    self.logger.warning("Returning default scores")
                    failed = True
                    scores = default_scores  # Use default scores for other errors
                    
  

        except ValueError as e:
            # Handle specific ValueError if AdaBoostClassifier fails due to poor performance
            if "BaseClassifier in AdaBoostClassifier ensemble is worse than random" in str(e):
                self.logger.warning(f"AdaBoostClassifier failed: {e}")
                self.logger.warning("Skipping AdaBoostClassifier due to poor base classifier performance.")
                
                # Set default scores if the AdaBoostClassifier fails
                scores = default_scores  # Use default scores
                
            else:
                self.logger.error(f"An unexpected ValueError occurred during cross-validation: {e}", exc_info=True)
                scores = default_scores  # Use default scores for other errors

        except Exception as e:
            # Catch any other general exceptions and log them
            self.logger.error(f"An unexpected error occurred during cross-validation: {e}", exc_info=True)
            scores = default_scores  # Use default scores if an error occurs

        # End the timer
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        if self.global_parameters.verbose >= 1:
            # Print a warning if the execution time exceeds the threshold
            if elapsed_time > time_threshold:
                self.logger.warning(f"Cross-validation took too long ({elapsed_time:.2f} seconds). Consider optimizing the parameters or reducing CV folds.")
            else:
                self.logger.info(f"Cross-validation for {method_name} completed in {elapsed_time:.2f} seconds.")
            
        
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
        best_pred_orig = current_algorithm.predict(self.X_test)  # exp
        
        

        project_score_save_class.update_score_log(
            
            ml_grid_object=ml_grid_object,
            scores=scores,
            best_pred_orig=best_pred_orig,
            current_algorithm=current_algorithm,
            method_name=method_name,
            pg=pg,
            start=start,
            n_iter_v=n_iter_v,
            failed=failed
        )
        
        # calculate metric for optimisation
        auc = metrics.roc_auc_score(self.y_test, best_pred_orig)
        
        self.grid_search_cross_validate_score_result = auc

    def _adjust_knn_parameters(self, parameter_space: Union[Dict, List[Dict]]):
        """
        Dynamically adjusts the 'n_neighbors' parameter for KNN-based models
        to prevent errors on small datasets during cross-validation.
        """
        # Smallest fold size will be n_samples * (n_splits-1)/n_splits
        # With RepeatedKFold, n_splits is at least 2. Smallest fold is 1/2 of data.
        n_splits = self.cv.get_n_splits()
        n_samples_in_fold = int(len(self.X_train) * (n_splits - 1) / n_splits)
        
        # Ensure n_samples_in_fold is at least 1
        n_samples_in_fold = max(1, n_samples_in_fold)

        def adjust_param(param_value):
            if is_skopt_space(param_value):
                # For skopt.space objects, adjust the upper bound
                new_high = min(param_value.high, n_samples_in_fold)
                new_low = min(param_value.low, new_high)
                param_value.high = new_high
                param_value.low = new_low
            elif isinstance(param_value, (list, np.ndarray)):
                # For lists, filter the values
                new_param_value = [n for n in param_value if n <= n_samples_in_fold]
                if not new_param_value:
                    return [n_samples_in_fold]
                return new_param_value
            return param_value

        if isinstance(parameter_space, list):
            for params in parameter_space:
                if 'n_neighbors' in params:
                    params['n_neighbors'] = adjust_param(params['n_neighbors'])
        elif isinstance(parameter_space, dict) and 'n_neighbors' in parameter_space:
            parameter_space['n_neighbors'] = adjust_param(parameter_space['n_neighbors'])



def dummy_auc() -> float:
    """Returns a constant AUC score of 0.5.

    This function is intended as a placeholder or for use in scenarios where
    a valid AUC score cannot be calculated but a value is required.
    
    Returns:
        float: A constant value of 0.5.
    """
    return 0.5

# Create a scorer using make_scorer
#dummy_auc_scorer = make_scorer(dummy_auc)




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
    if (min_val < 0 or max_val > 1):
        # Fit and transform the data
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        return X_train_scaled
    else:
        # If data is already scaled, return it as is
        return X_train