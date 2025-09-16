"""Global parameters for the ml_grid project.

This module defines a singleton class `GlobalParameters` to hold configuration
settings that are accessible throughout the application. It also includes a
custom scoring function for ROC AUC that handles cases with a single class.
"""

from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score


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

    Attributes:
        debug_level (int): Debug level, 0==minimal, 1,2,3,4
        knn_n_jobs (int): Number of jobs for knn, -1==all
        verbose (int): Verbose level for sklearn models
        rename_cols (bool): Rename cols of dataframes
        error_raise (bool): Raise errors from ml_grid
        random_grid_search (bool): Randomize search space for GridSearchCV
        sub_sample_param_space_pct (float): Percentage of param space to sub sample
        grid_n_jobs (int): Number of jobs for GridSearchCV
        time_limit_param (list): Time limit for GridSearchCV
        random_state_val (int): Random state value
        n_jobs_model_val (int): Number of jobs for models
        metric_list (dict): Dictionary of sklearn metrics to pass to GridSearchCV
        max_param_space_iter_value: hard limit on hyperparam search iterations.
        store_models (bool): Whether to save trained models to disk.
    """

    _instance = None

    # Class attributes with type hints
    debug_level: int
    knn_n_jobs: int
    verbose: int
    rename_cols: bool
    error_raise: bool
    random_grid_search: bool
    bayessearch: bool
    sub_sample_param_space_pct: float
    grid_n_jobs: int
    time_limit_param: List[int]
    random_state_val: int
    n_jobs_model_val: int
    max_param_space_iter_value: int
    store_models: bool
    metric_list: Dict[str, Union[str, Callable]]

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
        self.error_raise = False
        self.random_grid_search = False
        self.bayessearch = True
        self.sub_sample_param_space_pct = 0.0005  # 0.05==360
        self.grid_n_jobs = -1
        self.time_limit_param = [3]
        self.random_state_val = 1234
        self.n_jobs_model_val = -1
        self.max_param_space_iter_value = 10
        self.store_models = True

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
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

# Singleton instance
global_parameters = GlobalParameters()

    
    
