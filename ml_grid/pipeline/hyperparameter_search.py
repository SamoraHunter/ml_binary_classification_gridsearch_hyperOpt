import logging
import warnings
from typing import Any, Dict, List, Union

import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier
from ml_grid.model_classes.H2ODeepLearningClassifier import H2ODeepLearningClassifier
from ml_grid.model_classes.H2ODRFClassifier import H2ODRFClassifier
from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier
from ml_grid.model_classes.H2OGBMClassifier import H2OGBMClassifier
from ml_grid.model_classes.H2OGLMClassifier import H2OGLMClassifier
from ml_grid.model_classes.H2ONaiveBayesClassifier import H2ONaiveBayesClassifier
from ml_grid.model_classes.H2ORuleFitClassifier import H2ORuleFitClassifier
from ml_grid.model_classes.H2OStackedEnsembleClassifier import (
    H2OStackedEnsembleClassifier,
)
from ml_grid.model_classes.H2OXGBoostClassifier import H2OXGBoostClassifier
from ml_grid.model_classes.keras_classifier_class import KerasClassifierClass
#from ml_grid.model_classes.knn_wrapper_class import KNNWrapper
from ml_grid.model_classes.NeuralNetworkKerasClassifier import NeuralNetworkClassifier
from ml_grid.util.global_params import global_parameters
from ml_grid.util.validate_parameters import validate_parameters_helper


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
        cv: Any = None,
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
            cv (Any, optional): Cross-validation splitting strategy. Can be None, int,
                or a CV splitter. Defaults to None (no cross-validation).
        """
        self.algorithm = algorithm
        self.parameter_space = parameter_space
        self.method_name = method_name
        self.global_params = global_params
        self.sub_sample_pct = sub_sample_pct
        self.max_iter = max_iter
        self.ml_grid_object = ml_grid_object
        self.cv = cv

        if self.ml_grid_object is None:
            raise ValueError("ml_grid_object is required.")

        # Custom wrappers that might not be recognized by is_classifier
        custom_classifier_types = (
            #KNNWrapper,
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
            NeuralNetworkClassifier,  # type: ignore
            KerasClassifierClass,
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
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning
        )  # Suppress divide by zero warnings from NaiveBayes

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
            logger = logging.getLogger("ml_grid")
            gpu_devices = tf.config.experimental.list_physical_devices("GPU")
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            logger.warning(f"Could not configure GPU for TensorFlow: {e}")

    def run_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        """Executes the hyperparameter search.

        This method selects the search strategy (Grid, Random, or Bayesian) based
        on global parameters and runs the search on the provided training data.

        Args:
            X_train (pd.DataFrame): Training features with reset index.
            y_train (pd.Series): Training labels with reset index.

        Returns:
            BaseEstimator: The best estimator found during the search.
        """
        random_search = self.global_params.random_grid_search
        grid_n_jobs = self.global_params.grid_n_jobs
        bayessearch = self.global_params.bayessearch
        # Get main verbosity level for logging
        verbose = getattr(self.global_params, "verbose", 0)
        # Get specific verbosity for the search CV object, default to 0 (silent)
        search_verbose = getattr(self.global_params, "search_verbose", 0)

        # --- CRITICAL FIX for H2O multiprocessing ---
        # H2O models are not compatible with joblib's process-based parallelism.
        # We must detect if the algorithm is an H2O model and force n_jobs=1 for the search.
        h2o_models = (
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
        is_h2o_model = isinstance(self.algorithm, h2o_models)

        # Also limit n_jobs for Bayesian search and other specific wrappers to avoid issues.
        is_single_threaded_search = isinstance(
            self.algorithm, ( KerasClassifierClass, NeuralNetworkClassifier) #KNNWrapper,
        )

        if is_h2o_model or is_single_threaded_search or bayessearch:
            if verbose > 0:
                self.ml_grid_object.logger.info(
                    "Using n_jobs=1 to avoid pandas indexing issues in parallel processing"
                )
            grid_n_jobs = 1

        # Validate parameters - skip for Bayesian search as it uses different parameter format
        if not bayessearch:
            # Grid and Random search use standard sklearn parameter format (lists/arrays)
            parameters = validate_parameters_helper(
                algorithm_implementation=self.algorithm,
                parameters=self.parameter_space,
                ml_grid_object=self.ml_grid_object,
            )
        else:
            # Bayesian search uses skopt space objects (Integer, Real, Categorical)
            # These cannot go through standard validation
            parameters = self.parameter_space

        # Reset index to ensure clean integer indexing for CV splits
        # Keep as pandas to retain feature names
        if hasattr(X_train, "reset_index"):
            X_train_reset = X_train.reset_index(drop=True)
            if verbose > 1:
                self.ml_grid_object.logger.debug(
                    f"Reset X_train index. Shape: {X_train_reset.shape}"
                )
        else:
            X_train_reset = X_train

        if hasattr(y_train, "reset_index"):
            y_train_reset = y_train.reset_index(drop=True)
            if verbose > 1:
                self.ml_grid_object.logger.debug(
                    f"Reset y_train index. Shape: {y_train_reset.shape}"
                )
        else:
            y_train_reset = y_train

        # Verify data integrity
        if len(X_train_reset) != len(y_train_reset):
            raise ValueError(
                f"Length mismatch: X={len(X_train_reset)}, y={len(y_train_reset)}"
            )

        if verbose > 1:
            self.ml_grid_object.logger.debug(
                f"X_train type: {type(X_train_reset)}, shape: {X_train_reset.shape}"
            )
            self.ml_grid_object.logger.debug(
                f"y_train type: {type(y_train_reset)}, shape: {y_train_reset.shape}"
            )

        if bayessearch:
            # Bayesian Optimization
            grid = BayesSearchCV(
                estimator=self.algorithm,
                search_spaces=parameters,
                n_iter=self.max_iter,
                cv=self.cv,
                n_jobs=grid_n_jobs,
                verbose=search_verbose,
                error_score="raise",
            )

        elif random_search:
            grid = RandomizedSearchCV(
                self.algorithm,
                parameters,
                verbose=search_verbose,
                cv=self.cv,
                n_jobs=grid_n_jobs,
                n_iter=self.max_iter,
                error_score="raise",
            )
        else:
            # Grid Search
            grid = GridSearchCV(
                self.algorithm,
                parameters,
                verbose=search_verbose,
                cv=self.cv,
                n_jobs=grid_n_jobs,
                error_score="raise",
            )

        if verbose > 0:
            self.ml_grid_object.logger.info(
                f"Starting hyperparameter search with {len(X_train_reset)} samples"
            )

        # Fit the grid search with pandas DataFrames/Series (retains feature names)
        grid.fit(X_train_reset, y_train_reset)

        best_model = grid.best_estimator_
        return best_model
