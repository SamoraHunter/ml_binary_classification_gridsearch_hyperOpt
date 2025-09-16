import inspect
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.exceptions import ConvergenceWarning
from skopt import BayesSearchCV
from sklearn.base import is_classifier, BaseEstimator

from ml_grid.util.validate_parameters import validate_parameters_helper
from ml_grid.util.global_params import global_parameters
from ml_grid.model_classes.knn_wrapper_class import KNNWrapper
from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier


class HyperparameterSearch:
    """Orchestrates hyperparameter search using GridSearchCV, RandomizedSearchCV, or BayesSearchCV."""

    algorithm: BaseEstimator
    """The scikit-learn compatible estimator instance."""

    parameter_space: Union[Dict, List[Dict]]
    """The hyperparameter search space."""

    method_name: str
    """The name of the algorithm."""

    global_params: global_parameters
    """A reference to the global parameters singleton instance."""

    sub_sample_pct: int
    """
    Percentage of the parameter space to sample for randomized search.
    Defaults to 100.
    """

    max_iter: int
    """
    The maximum number of iterations for randomized or Bayesian search.
    Defaults to 100.
    """

    ml_grid_object: Any
    """The main pipeline object containing data and other parameters."""

    def __init__(
        self,
        algorithm: BaseEstimator,
        parameter_space: Union[Dict, List[Dict]],
        method_name: str,
        global_params: Any,
        sub_sample_pct: int = 100,
        max_iter: int = 100,
        ml_grid_object: Any = None,
    ):
        """Initializes the HyperparameterSearch class.

        Args:
            algorithm (BaseEstimator): The scikit-learn compatible estimator instance.
            parameter_space (Union[Dict, List[Dict]]): The hyperparameter search space.
            method_name (str): The name of the algorithm.
            global_params (Any): The global parameters object.
            sub_sample_pct (int, optional): Percentage of the parameter space to sample for
                randomized search. Defaults to 100.
            max_iter (int, optional): The maximum number of iterations for randomized or
                Bayesian search. Defaults to 100.
            ml_grid_object (Any, optional): The main pipeline object containing data and
                other parameters. Defaults to None.
        """
        self.algorithm = algorithm
        self.parameter_space = parameter_space
        self.method_name = method_name
        self.global_params = global_params
        self.sub_sample_pct = sub_sample_pct
        self.max_iter = max_iter
        self.ml_grid_object = ml_grid_object

        if self.ml_grid_object is None:
            raise ValueError("ml_grid_object is required.")

        # Custom wrappers that might not be recognized by is_classifier
        custom_classifier_types = (
            KNNWrapper,
            H2OAutoMLClassifier,
            kerasClassifier_class,
        )

        # Check if it's a valid classifier
        is_valid = (
            is_classifier(self.algorithm)
            or isinstance(self.algorithm, custom_classifier_types)
            or (hasattr(self.algorithm, "fit") and hasattr(self.algorithm, "predict"))
        )

        if not is_valid:
            raise ValueError(
                f"The provided algorithm is not a valid classifier. "
                f"Received type: {type(self.algorithm)}"
            )

        # Configure warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Configure GPUs if applicable
        if (
            "keras" in method_name.lower()
            or "xgb" in method_name.lower()
            or "catboost" in method_name.lower()
        ):
            self._configure_gpu()

    def _configure_gpu(self) -> None:
        """Configures TensorFlow to use GPU with memory growth enabled."""
        try:
            gpu_devices = tf.config.experimental.list_physical_devices("GPU")
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Could not configure GPU for TensorFlow: {e}")

    def run_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        """Executes the hyperparameter search.

        This method selects the search strategy (Grid, Random, or Bayesian) based
        on global parameters and runs the search on the provided training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            BaseEstimator: The best estimator found during the search.
        """
        random_search = self.global_params.random_grid_search
        grid_n_jobs = self.global_params.grid_n_jobs
        bayessearch = self.global_params.bayessearch

        # Limit n_jobs for GPU-heavy methods to avoid memory issues
        gpu_heavy_models = (KNNWrapper, kerasClassifier_class)
        if bayessearch and isinstance(self.algorithm, gpu_heavy_models):
            grid_n_jobs = 1

        if not bayessearch:
            # Validate parameters
            parameters = validate_parameters_helper(
                algorithm_implementation=self.algorithm,
                parameters=self.parameter_space,
                ml_grid_object=self.ml_grid_object
            )
        else:
            parameters = self.parameter_space

        if bayessearch:
            # Bayesian Optimization
            grid = BayesSearchCV(
                estimator=self.algorithm,
                search_spaces=parameters,
                n_iter=self.max_iter,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                verbose=1,
                error_score="raise",
            )

        elif random_search:
            n_iter = min(
                self.max_iter,
                max(2, int(len(ParameterGrid(parameters)) * self.sub_sample_pct / 100)),
            )

            grid = RandomizedSearchCV(
                self.algorithm,
                parameters,
                verbose=1,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                n_iter=n_iter,
                error_score="raise",
            )
        else:
            grid = GridSearchCV(
                self.algorithm,
                parameters,
                verbose=1,
                cv=[(slice(None), slice(None))],
                n_jobs=grid_n_jobs,
                error_score=np.nan,
            )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        return best_model
