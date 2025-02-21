import time
import traceback
import warnings

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from numpy import absolute, mean, std
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from IPython.display import display
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
from ml_grid.util.bayes_utils import calculate_combinations
from skopt.space import Categorical

class grid_search_crossvalidate:

    def __init__(
        self,
        algorithm_implementation,
        parameter_space,
        method_name,
        ml_grid_object,
        sub_sample_parameter_val=100,
    ):  # kwargs**
        #

        warnings.filterwarnings("ignore")

        warnings.filterwarnings("ignore", category=FutureWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        warnings.filterwarnings("ignore", category=UserWarning)

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        if self.verbose < 8:
            print(f"Clearing ")
            clear_output(wait=True)

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        random_grid_search = self.global_params.random_grid_search

        self.sub_sample_parameter_val = sub_sample_parameter_val

        grid_n_jobs = self.global_params.grid_n_jobs

        if "keras" in method_name.lower(): # kerasClassifier_class
            grid_n_jobs = 1
            gpu_devices = tf.config.experimental.list_physical_devices("GPU")
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)

        if "XGBClassifier".lower() in str(algorithm_implementation).lower():
            grid_n_jobs = -1

        if "CatBoostClassifier".lower() in method_name.lower(): #CatBoostClassifier
            grid_n_jobs = 1 # needs to write to directory

        

        self.metric_list = self.global_params.metric_list

        self.error_raise = self.global_params.error_raise

        if self.verbose >= 3:
            print(f"crossvalidating {method_name}")

        self.global_parameters = global_parameters()

        self.ml_grid_object_iter = ml_grid_object

        self.X_train = self.ml_grid_object_iter.X_train

        self.y_train = self.ml_grid_object_iter.y_train

        self.X_test = self.ml_grid_object_iter.X_test

        self.y_test = self.ml_grid_object_iter.y_test

        self.X_test_orig = self.ml_grid_object_iter.X_test_orig

        self.y_test_orig = self.ml_grid_object_iter.y_test_orig

        max_param_space_iter_value = 10 # hard limit on param space exploration

        if "svc" in method_name.lower():
            self.X_train = scale_data(self.X_train)
            self.X_test = scale_data(self.X_test)

        self.cv = RepeatedKFold(
            n_splits=max(2, min(len(self.X_train), 2) + 1), 
            n_repeats=2, 
            random_state=1
        )

        start = time.time()

        current_algorithm = algorithm_implementation

        parameters = parameter_space
        if(self.global_params.bayessearch is False):
            n_iter_v = np.nan
        else:
            n_iter_v = 2
        #     if(sub_sample_param_space):
        #         sub_sample_param_space_n = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space)))
        #         parameter_space random.sample(ParameterGrid(parameter_space), sub_sample_param_space_n)

        # Grid search over hyperparameter space, randomised.

        if(ml_grid_object.verbose >= 3):
            print("algorithm_implementation: ", algorithm_implementation, " type: ", type(algorithm_implementation), )
        
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
        #         print("warn n_iter_v < 2")
        #         n_iter_v = 2
        #     if n_iter_v > max_param_space_iter_value:
        #         print(f"Warn n_iter_v > max_param_space_iter_value, setting {max_param_space_iter_value}")
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
        #print(f"Approximate number of combinations: {approx_combinations}")
        # print(pg) 
 
        if (random_grid_search and n_iter_v > 100000) or (
            random_grid_search == False and pg > 100000
        ):
            print("grid too large", str(pg), str(n_iter_v))
            print("Warning grid too large, ", str(pg))
            # raise Exception("grid too large", str(pg))

        if self.global_parameters.verbose >= 1:
            if random_grid_search:
                print(
                    f"Randomized parameter grid size for {current_algorithm} \n : Full: {pg}, (mean * {self.sub_sample_param_space_pct}): {self.sub_sample_parameter_val}, current: {n_iter_v} "
                )

            else:
                print(f"parameter grid size: Full: {pg}")

        #grid.fit(self.X_train, self.y_train)
        if self.global_parameters.bayessearch:
            n_iter_v = pg + 2
        else:
            n_iter_v = int(len(ParameterGrid(parameter_space))) + 2 #review relevance and value

        if self.sub_sample_parameter_val < n_iter_v:
            n_iter_v = self.sub_sample_parameter_val
        if n_iter_v < 2:
            print("warn n_iter_v < 2")
            n_iter_v = 2
        if n_iter_v > max_param_space_iter_value:
            print(f"Warn n_iter_v > max_param_space_iter_value, setting {max_param_space_iter_value}")
            n_iter_v = max_param_space_iter_value
        print("n_iter_v = ", n_iter_v)

        # Instantiate and run the hyperparameter grid/random search
        search = HyperparameterSearch(
            algorithm=current_algorithm,
            parameter_space=parameter_space,
            method_name=method_name,
            global_params=self.global_parameters,
            sub_sample_pct=self.sub_sample_param_space_pct,  # Explore 50% of the parameter space
            max_iter=n_iter_v,         # Maximum iterations for randomized search
            ml_grid_object=ml_grid_object
        )

        if self.global_parameters.verbose >= 3:
            print("Running hyperparameter search")
        
        try:    
            
            current_algorithm = search.run_search(self.X_train, self.y_train)
            
        except XGBoostError as e:
            if 'cudaErrorMemoryAllocation' in str(e) or 'std::bad_alloc' in str(e):
                print("GPU memory error detected, falling back to CPU...")
                 
                 # Change the tree_method in parameter_space dynamically
                for param_dict in parameter_space:
                    if 'tree_method' in param_dict:
                        if(self.global_params.bayessearch):
                            param_dict['tree_method'] = Categorical(['hist'])
                        else:
                            param_dict['tree_method'] = ["hist"]
                 
                    search = HyperparameterSearch(
                        algorithm=current_algorithm,
                        parameter_space=parameter_space,
                        method_name=method_name,
                        global_params=self.global_parameters,
                        sub_sample_pct=self.sub_sample_param_space_pct,  # Explore 50% of the parameter space
                        max_iter=n_iter_v,         # Maximum iterations for randomized search
                        ml_grid_object=ml_grid_object
                    )
                    # Try again with non gpu method. 
                    current_algorithm = search.run_search(self.X_train, self.y_train)
            else: 
                print("unknown xgb error")
                print(e)
            
        except Exception as e:
            print(e)
            print("Failed to run search in gridsearch cross validate")
            


        if self.global_parameters.verbose >= 3:
            print("Fitting final model")
        #current_algorithm = grid.best_estimator_
        current_algorithm.fit(self.X_train, self.y_train)

        metric_list = self.metric_list

        # Catch only one class present AUC not defined:
        
        #dummy_auc_scorer = make_scorer(dummy_auc)
        if len(np.unique(self.y_train)) < 2:
            raise ValueError("Only one class present in y_train. ROC AUC score is not defined in that case. grid_search_cross_validate>>>cross_validate")

        if self.global_parameters.verbose >= 1:
            print("Getting cross validation scores")
            print(self.X_train.shape, self.y_train.shape)
            print("y_train value counts:")
            print(self.y_train.value_counts())

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

        try:
            # Perform the cross-validation
            scores = cross_validate(
                current_algorithm,
                self.X_train,
                self.y_train,
                scoring=self.metric_list,
                cv=self.cv,
                n_jobs=grid_n_jobs,  # Full CV on final best model
                pre_dispatch=80,
                error_score='raise',  # Raise error if cross-validation fails
            )
            
            
        except XGBoostError as e:
            if 'cudaErrorMemoryAllocation' in str(e) or 'std::bad_alloc' in str(e):
                print("GPU memory error detected, falling back to CPU...")
                current_algorithm.set_params(tree_method='hist') 
                
                try:
                    scores = cross_validate(
                        current_algorithm,
                        self.X_train,
                        self.y_train,
                        scoring=self.metric_list,
                        cv=self.cv,
                        n_jobs=grid_n_jobs,  # Full CV on final best model
                        pre_dispatch=80,
                        error_score='raise',  # Raise error if cross-validation fails
                    )
                except Exception as e:
                    print(f"An unexpected error occurred during cross-validation attempt 2: {e}")
                    scores = default_scores  # Use default scores for other errors
                    
  

        except ValueError as e:
            # Handle specific ValueError if AdaBoostClassifier fails due to poor performance
            if "BaseClassifier in AdaBoostClassifier ensemble is worse than random" in str(e):
                print(f"AdaBoostClassifier failed: {e}")
                print("Skipping AdaBoostClassifier due to poor base classifier performance.")
                
                # Set default scores if the AdaBoostClassifier fails
                scores = default_scores  # Use default scores
                
            else:
                print(f"An unexpected error occurred during cross-validation: {e}")
                scores = default_scores  # Use default scores for other errors

        except Exception as e:
            # Catch any other general exceptions and log them
            print(f"An error occurred during cross-validation: {e}")
            scores = default_scores  # Use default scores if an error occurs

        # End the timer
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        if self.global_parameters.verbose >= 1:
            # Print a warning if the execution time exceeds the threshold
            if elapsed_time > time_threshold:
                print(f"Warning: Cross-validation took too long ({elapsed_time:.2f} seconds). Consider optimizing the parameters or reducing CV folds.")
            else:
                print(f"Cross-validation {method_name} completed in {elapsed_time:.2f} seconds.")
            
        
        current_algorithm_scores = scores
        #     scores_tuple_list.append((method_name, current_algorithm_scores, grid))

        if self.global_parameters.verbose >= 4:

            debug_print_statements_class(scores).debug_print_scores()

        plot_auc = False
        if plot_auc:
            # This was passing a classifier trained on the test dataset....
            print(" ")

            # plot_auc_results(current_algorithm, self.X_test_orig[self.X_train.columns], self.y_test_orig, self.cv)
            # plot_auc_results(grid.best_estimator_, X_test_orig, self.y_test_orig, cv)

        #         this should be x_test...?
        best_pred_orig = current_algorithm.predict(self.X_test)  # exp

        project_score_save_class.update_score_log(
            self=self,
            ml_grid_object=self.ml_grid_object_iter,
            scores=scores,
            best_pred_orig=best_pred_orig,
            current_algorithm=current_algorithm,
            method_name=method_name,
            pg=pg,
            start=start,
            n_iter_v=n_iter_v,
        )
        
        # return metric for optimisation
        auc = metrics.roc_auc_score(self.y_test, best_pred_orig)

        return auc



def dummy_auc():
    """
    Dummy function to return a constant AUC score of 0.5.
    
    Parameters:
    - y_true : array-like of shape (n_samples,)
        True binary labels.
    - y_pred : array-like of shape (n_samples,)
        Target scores, can either be probability estimates or confidence values.
        
    Returns:
    - auc_score : float
        Constant AUC score of 0.5.
    """
    return 0.5

# Create a scorer using make_scorer
#dummy_auc_scorer = make_scorer(dummy_auc)




def scale_data(X_train):
    """
    Scale the data to [0, 1] range if it's not already scaled.
    
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
