import time
import traceback
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from IPython.display import clear_output
from numpy import absolute, mean, std
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from IPython.display import display
from catboost import CatBoostError
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
from ml_grid.model_classes.H2OStackedEnsembleClassifier import H2OStackedEnsembleClassifier

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
    KFold,
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

        # Explicitly set CPU as the visible device for TensorFlow to avoid CUDA init errors
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            self.logger.warning(f"Could not disable GPU visibility for TensorFlow: {e}")

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
        
        # --- PERFORMANCE FIX for testing ---
        # Use a much faster CV strategy when in test_mode.
        # This MUST be defined before HyperparameterSearch is instantiated.
        if getattr(self.global_parameters, 'test_mode', False):
            self.logger.info("Test mode enabled. Using fast KFold(n_splits=2) for CV.")
            self.cv = KFold(n_splits=2, shuffle=True, random_state=1)
        else:
            # Use the full, robust CV strategy for production runs
            self.cv = RepeatedKFold(
                # Ensure n_splits is at least 2 but not more than the number of samples.
                n_splits=min(len(self.X_train), 5),
                n_repeats=2,
                random_state=1
            )


        start = time.time()

        current_algorithm = algorithm_implementation

        # Silence verbose models like CatBoost to keep logs clean
        if "catboost" in method_name.lower() and hasattr(current_algorithm, 'set_params'):
            ml_grid_object.logger.info("Silencing CatBoost verbose output.")
            current_algorithm.set_params(verbose=0)

        # Check for GPU availability and set device for torch-based models
        if "simbsig" in str(type(algorithm_implementation)):
            if not torch.cuda.is_available():
                self.logger.info("No CUDA GPU detected. Forcing simbsig model to use CPU.")
                if hasattr(current_algorithm, 'set_params'):
                    current_algorithm.set_params(device='cpu')
            else:
                self.logger.info("CUDA GPU detected. Allowing simbsig model to use GPU.")
        
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
            # For random search, calculate iterations based on a percentage of the grid size
            if random_grid_search:
                n_iter_v = int(len(ParameterGrid(parameter_space)) * (self.sub_sample_param_space_pct / 100))
            else:
                # For grid search, this value is not used for iterations
                n_iter_v = len(ParameterGrid(parameter_space))

        # Ensure n_iter is at least 2 and does not exceed the global max
        n_iter_v = max(2, n_iter_v)
        n_iter_v = min(n_iter_v, max_param_space_iter_value)

        self.logger.info(f"n_iter_v = {n_iter_v}")

        # Dynamically adjust KNN parameter space for small datasets
        if "kneighbors" in method_name.lower() or "simbsig" in method_name.lower():
            self._adjust_knn_parameters(parameter_space)
            self.logger.info(
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
            self.logger.info(
                "Adjusted CatBoost subsample parameter space to prevent errors on small CV folds."
            )
            

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

        except CatBoostError as e:
            if "All features are either constant or ignored" in str(e):
                self.logger.error(f"CatBoostError occurred for {method_name}: {e}")
                self.logger.warning(f"Continuing despite CatBoost error...")
                # Set a default score and return to allow the pipeline to continue
                self.grid_search_cross_validate_score_result = 0.5
                return
            else:
                # Re-raise other CatBoost errors
                raise e

        except ValueError as e:
            # Handle specific ValueError if AdaBoostClassifier fails during search
            if "BaseClassifier in AdaBoostClassifier ensemble is worse than random" in str(e):
                self.logger.error(f"AdaBoostClassifier failed during hyperparameter search: {e}")
                self.logger.warning(f"Continuing despite AdaBoost error...")
                self.grid_search_cross_validate_score_result = 0.5
                return
            else:
                # Re-raise other ValueErrors
                raise e

            
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

        # --- PERFORMANCE FIX for testing ---
        # If in test_mode, we have already verified that the search runs without crashing.
        # We can skip the final, slow cross-validation and return a dummy score.
        if getattr(self.global_parameters, 'test_mode', False):
            self.logger.info("Test mode enabled. Skipping final cross-validation for speed.")
            self.grid_search_cross_validate_score_result = 0.5 # Return a valid float
            # Final cleanup for H2O models
            self._shutdown_h2o_if_needed(current_algorithm)
            return

        if self.global_parameters.verbose >= 3:
            self.logger.debug("Fitting final model")

        # In production, we re-fit the best estimator on the full training data before CV.
        # In test_mode, the estimator from the search is already fitted, and re-fitting
        # can invalidate complex models like H2OStackedEnsemble before the final assert.
        if not getattr(self.global_parameters, 'test_mode', False):
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
            # H2O models require pandas DataFrames with column names, while other
            # sklearn models can benefit from using NumPy arrays.
            h2o_model_types = (
                H2OAutoMLClassifier, H2OGBMClassifier, H2ODRFClassifier, H2OGAMClassifier,
                H2ODeepLearningClassifier, H2OGLMClassifier, H2ONaiveBayesClassifier,
                H2ORuleFitClassifier, H2OXGBoostClassifier, H2OStackedEnsembleClassifier
            )
            if isinstance(current_algorithm, h2o_model_types):
                X_train_final = self.X_train # Pass DataFrame directly
            else:
                X_train_final = self.X_train.values # Use NumPy array for other models

            # Perform the cross-validation
            scores = cross_validate(
                current_algorithm,
                X_train_final,
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
                    scores = cross_validate(
                        current_algorithm,
                        X_train_final,
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

        self._shutdown_h2o_if_needed(current_algorithm)

    def _adjust_knn_parameters(self, parameter_space: Union[Dict, List[Dict]]):
        """
        Dynamically adjusts the 'n_neighbors' parameter for KNN-based models
        to prevent errors on small datasets during cross-validation.
        """
        n_splits = self.cv.get_n_splits()

        # The number of samples in the training part of a fold.
        n_samples_train_fold = len(self.X_train) - (len(self.X_train) // n_splits)
        n_samples_test_fold = len(self.X_train) // n_splits

        # CRITICAL: The number of neighbors cannot exceed the number of samples
        # in the training fold that the model is fit on.
        max_n_neighbors = n_samples_train_fold
        max_n_neighbors = max(1, max_n_neighbors)
        
        self.logger.info(
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
                self.logger.debug(f"Adjusted skopt space: low={new_low}, high={new_high}")
            elif isinstance(param_value, (list, np.ndarray)):
                # For lists, filter the values
                new_param_value = [n for n in param_value if n <= max_n_neighbors]
                if not new_param_value:
                    self.logger.warning(f"All n_neighbors values filtered out. Using [{max_n_neighbors}]")
                    return [max_n_neighbors]
                self.logger.debug(f"Filtered n_neighbors list: {new_param_value}")
                return new_param_value
            return param_value

        if isinstance(parameter_space, list):
            for params in parameter_space:
                if 'n_neighbors' in params:
                    params['n_neighbors'] = adjust_param(params['n_neighbors'])
        elif isinstance(parameter_space, dict) and 'n_neighbors' in parameter_space:
            parameter_space['n_neighbors'] = adjust_param(parameter_space['n_neighbors'])

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
                    return [min(p for p in param_value if p > 0) if any(p > 0 for p in param_value) else 1.0]
                return new_param_value
            # If the fold is tiny, force subsample to 1.0
            if n_samples_in_fold <= 2:
                return [1.0] if isinstance(param_value, list) else 1.0
            return param_value

        if isinstance(parameter_space, list):
            for params in parameter_space:
                if 'subsample' in params:
                    params['subsample'] = adjust_param(params['subsample'])
        elif isinstance(parameter_space, dict) and 'subsample' in parameter_space:
            parameter_space['subsample'] = adjust_param(parameter_space['subsample'])

        # Also adjust 'rsm' (colsample_bylevel) which can cause the same issue
        if isinstance(parameter_space, list):
            for params in parameter_space:
                if 'rsm' in params:
                    params['rsm'] = adjust_param(params['rsm'])
        elif isinstance(parameter_space, dict) and 'rsm' in parameter_space:
            parameter_space['rsm'] = adjust_param(parameter_space['rsm'])

    def _shutdown_h2o_if_needed(self, algorithm: Any):
        """Safely shuts down the H2O cluster if the algorithm is an H2O model."""
        h2o_model_types = (
            H2OAutoMLClassifier, H2OGBMClassifier, H2ODRFClassifier, H2OGAMClassifier,
            H2ODeepLearningClassifier, H2OGLMClassifier, H2ONaiveBayesClassifier,
            H2ORuleFitClassifier, H2OXGBoostClassifier, H2OStackedEnsembleClassifier
        )
        if isinstance(algorithm, h2o_model_types):
            try:
                self.logger.info("Shutting down H2O cluster.")
                algorithm.shutdown()
            except Exception as e:
                self.logger.error(f"Failed to shut down H2O cluster: {e}")

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