"""Global parameters for the ml_grid project.

This module defines a singleton class `GlobalParameters` to hold configuration
settings that are accessible throughout the application. It also includes a
custom scoring function for ROC AUC that handles cases with a single class.
"""

from typing import Any, Callable, Dict, List, Union
import logging

import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score


def custom_roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates ROC AUC score, handling cases with only one class in y_true.

    If `y_true` contains fewer than two unique classes, ROC AUC is undefined.
    In such cases, this function returns np.nan.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Target scores.

    Returns:
        float: The ROC AUC score, or np.nan if the score is undefined.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan  # Return NaN if only one class is present
    else:
        return roc_auc_score(y_true, y_pred)


class GlobalParameters:
    """A singleton class to manage global configuration parameters for ml_grid.

    This class holds all global settings, making them accessible from any part
    of the application. To change a parameter, access the singleton instance
    `global_parameters` and set the attribute directly, or use the
    `update_parameters` method.

    Example:
        from ml_grid.util.global_params import global_parameters
        global_parameters.verbose = 2
        global_parameters.update_parameters(bayessearch=False, random_grid_search=True)
    """

    _instance = None

    # Class attributes with type hints
    debug_level: int
    """The verbosity level for debugging. Not widely used. Defaults to 0."""
    knn_n_jobs: int
    """The number of parallel jobs to run for KNN algorithms. -1 means using all available processors. Defaults to -1."""
    verbose: int
    """Controls the verbosity of output during the pipeline run. Higher values produce more detailed logs. Defaults to 0."""
    rename_cols: bool
    """If True, renames DataFrame columns to remove special characters (e.g., '[, ], <') that can cause issues with some models like XGBoost. Defaults to True."""
    error_raise: bool
    """If True, the pipeline will stop and raise an exception if an error occurs during model training or evaluation. If False, it will log the error and continue. Defaults to False."""
    random_grid_search: bool
    """If True and `bayessearch` is False, uses `RandomizedSearchCV` instead of `GridSearchCV`. Defaults to False."""
    bayessearch: bool
    """If True, uses `BayesSearchCV` from `scikit-optimize` for hyperparameter tuning, which can be more efficient than grid or random search. Defaults to True."""
    sub_sample_param_space_pct: float
    """The percentage of the total parameter space to sample when using `RandomizedSearchCV`. For example, 0.1 means 10% of the combinations will be tried. Defaults to 0.0005."""
    grid_n_jobs: int
    """The number of jobs to run in parallel for hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`, `BayesSearchCV`). -1 means using all available processors. Defaults to -1."""
    time_limit_param: List[int]
    """A parameter for future use, intended to set time limits on model fitting. Currently not implemented. Defaults to [3]."""
    random_state_val: int
    """A seed value for random number generation to ensure reproducibility across runs. Defaults to 1234."""
    n_jobs_model_val: int
    """The number of parallel jobs for models that support it (e.g., RandomForest). -1 means using all available processors. Defaults to -1."""
    max_param_space_iter_value: int
    """A hard limit on the number of parameter combinations to evaluate in `RandomizedSearchCV` or `BayesSearchCV`. Prevents excessively long run times. Defaults to 10."""
    store_models: bool
    """Whether to save trained models to disk. Defaults to True."""
    metric_list: Dict[str, Union[str, Callable]]
    """A dictionary of scoring metrics to evaluate models during cross-validation. Keys are metric names and values are scikit-learn scorer strings or callable objects."""
    use_embedding: bool
    """Whether to use embedding for feature transformation. Defaults to False."""
    embedding_method: str
    """The embedding method to use ("svd", "pca", "nmf", "lda", "random_gaussian", "random_sparse", "select_kbest_f", "select_kbest_mi"). Defaults to None."""
    embedding_dim: int
    """The dimensionality of the embedding space. Defaults to None."""
    scale_features_before_embedding: bool
    """Whether to scale features before applying embedding. Defaults to False."""
    cache_embeddings: bool
    """Whether to cache computed embeddings for reuse. Defaults to False."""
    n_iter: int
    """Number of iterations for randomized/Bayesian search. Defaults to 2."""
    h2o_show_progress: bool
    """If True, shows H2O progress bars. Defaults to False."""
    search_verbose: int
    """Verbosity level for the search object (GridSearchCV, etc.). Defaults to 0."""
    force_second_cv: bool
    """If True, forces a second cross-validation run even if cached results are available. Defaults to False."""
    model_eval_time_limit: int
    """The time limit in seconds for a single model evaluation. Defaults to None (no limit)."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "GlobalParameters":
        """Creates a new instance if one does not already exist (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(GlobalParameters, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, debug_level: int = 0, knn_n_jobs: int = -1) -> None:
        """Initializes the GlobalParameters instance.

        This method sets the default values for all global parameters. The
        `_initialized` flag prevents re-initialization on subsequent calls.

        Args:
            debug_level (int, optional): The initial debug level. Defaults to 0.
            knn_n_jobs (int, optional): The number of jobs for KNN. Defaults to -1.
        """
        if self._initialized:
            return
        self._initialized = True

        self.debug_level = debug_level
        self.knn_n_jobs = knn_n_jobs
        self.verbose = 0
        self.rename_cols = True
        self.error_raise = True
        self.random_grid_search = False
        self.bayessearch = True
        self.sub_sample_param_space_pct = 0.0005  # 0.05==360
        self.grid_n_jobs = -1
        self.time_limit_param = [3]
        self.random_state_val = 1234
        self.n_jobs_model_val = -1
        self.max_param_space_iter_value = 10
        self.store_models = False
        self.use_embedding = False
        self.embedding_method = None  # "svd", "pca", "nmf", "lda", "random_gaussian", "random_sparse", "select_kbest_f", "select_kbest_mi"
        self.embedding_dim = None
        self.scale_features_before_embedding = False
        self.cache_embeddings = False
        self.n_iter = 2
        self.h2o_show_progress = False
        self.search_verbose = 0
        self.force_second_cv = False
        self.model_eval_time_limit = None

        custom_scorer = make_scorer(custom_roc_auc_score)
        self.metric_list = {
            "auc": custom_scorer,
            "f1": "f1",
            "accuracy": "accuracy",
            "recall": "recall",
        }

    def update_parameters(self, **kwargs: Any) -> None:
        """Updates global parameters at runtime.

        Args:
            **kwargs (Any): Key-value pairs of parameters to update.

        Raises:
            AttributeError: If a key in kwargs is not a valid parameter.
        """
        logger = logging.getLogger("ml_grid")
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated GlobalParameter: {key} = {value}")
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{key}'"
                )


# Singleton instance
global_parameters = GlobalParameters()
