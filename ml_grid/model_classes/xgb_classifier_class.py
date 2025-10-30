"""XGBoost Classifier.

This module contains the XGB_class_class, which is a configuration
class for the XGBClassifier. It provides parameter spaces for
grid search and Bayesian optimization.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from ml_grid.util import param_space
from skopt.space import Real, Categorical, Integer
from ml_grid.util.global_params import global_parameters
import logging

logging.getLogger('ml_grid').debug("Imported XGB class")

class XGB_class_class:
    """XGBoost classifier with support for Bayesian and Grid Search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        """Initializes the XGB_class_class.

        The XGB_class_class wraps the XGBoost classifier algorithm, allowing for
        easy configuration and use within a grid search or Bayesian optimization
        framework by setting up a customizable parameter space.

        Args:
            X (Optional[pd.DataFrame]): Feature matrix for training. Defaults to None.
            y (Optional[pd.Series]): Target vector for training. Defaults to None.
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this).
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y

        # Initialize the algorithm implementation using XGBClassifier
        self.algorithm_implementation: xgb.XGBClassifier = xgb.XGBClassifier()
        self.method_name: str = "XGBClassifier"

        # Initialize the parameter space handler
        self.parameter_vector_space: param_space.ParamSpace = param_space.ParamSpace(
            parameter_space_size
        )

        self.parameter_space: Dict[str, Any]

        # Patch max_bin dynamically for compatibility
        def patch_max_bin(param_value: Any) -> Union[int, Real, Integer, Any]:
            """Ensures the 'max_bin' parameter is >= 2.

            XGBoost's 'max_bin' parameter must be at least 2. This function
            patches the provided value to meet this requirement, handling integers
            and skopt space objects.

            Args:
                param_value (Any): The original parameter value, which can be an
                    integer or a skopt space object (Real, Integer).

            Returns:
                Union[int, Real, Integer, Any]: The patched parameter value,
                ensuring it is >= 2.
            """
            if isinstance(param_value, int):
                return max(2, param_value)
            elif hasattr(param_value, "rvs"):  # For sampled values (e.g., skopt spaces)
                return Real(2, param_value.high, prior=param_value.prior) if isinstance(param_value, Real) else Integer(2, param_value.high)
            else:
                return param_value

        # Set up the parameter space based on the selected optimization method
        if global_parameters.bayessearch:
            # Bayesian Optimization: Define parameter space using Real and Categorical
            self.parameter_space = {
                "objective": Categorical(["binary:logistic"]),  # Objective function for binary classification
                "booster": Categorical(["gbtree", "gblinear", "dart"]),  # Type of boosting model
                "gamma": Real(1e-5, 1e-2, prior="log-uniform"),  # Regularization parameter
                "grow_policy": Categorical(["depthwise", "lossguide"]),  # Tree growth policy
                "learning_rate": Real(1e-5, 1e-2, prior="log-uniform"),  # Learning rate
                "max_bin": Integer(2, 100),
                "max_depth": Integer(2, 50),
                "max_leaves": Integer(2, 100),
                "min_child_weight": Real(1e-5, 1e-2, prior="log-uniform"),
                "n_estimators": Integer(50, 500),
                "n_jobs": Categorical([-1]),  # Number of parallel threads to use for training
                "random_state": Categorical([None]),  # Random state for reproducibility
                "reg_alpha": Real(1e-5, 1e-2, prior="log-uniform"),
                "reg_lambda": Real(1e-5, 1e-2, prior="log-uniform"),
                "sampling_method": Categorical(["uniform"]),  # Sampling method during training
                "verbosity": Categorical([0]),  # Verbosity level during training
                "tree_method": Categorical(["auto"])
            }

                # Future use parameters for Bayesian optimization
                # "use_label_encoder": Categorical([True, False]),  # Use label encoder
                # "base_score": Real(0.0, 1.0, "uniform"),  # Base score for predictions
                # "callbacks": [None],  # Custom callbacks for training
                # "colsample_bylevel": Real(0.5, 1, "uniform"),  # Column sampling by level
                # "colsample_bynode": Real(0.5, 1, "uniform"),  # Column sampling by node
                # "colsample_bytree": Real(0.5, 1, "uniform"),  # Column sampling by tree
                # "early_stopping_rounds": Categorical([None]),  # Early stopping for boosting rounds
                # "enable_categorical": Categorical([True, False]),  # Enable categorical variables (if needed)
                # "eval_metric": Categorical([None]),  # Evaluation metric (optional)
                # "gpu_id": Categorical([None]),  # GPU id to use for training
                # "importance_type": Categorical(["weight", "gain", "cover"]),  # Type of feature importance calculation
                # "interaction_constraints": Categorical([None]),  # Constraints for feature interaction
                # "max_cat_to_onehot": Real(1, 100, "uniform"),  # Max categories for one-hot encoding
                # "max_delta_step": Real(0, 10, "uniform"),  # Max delta step for optimization
                # "monotone_constraints": Categorical([None]),  # Constraints for monotonicity in predictions
                # "num_parallel_tree": Real(1, 10, "uniform"),  # Number of parallel trees in boosting
                # "predictor": Categorical(["cpu_predictor", "gpu_predictor"]),  # Type of predictor (e.g., 'gpu_predictor')
                # "scale_pos_weight": Real(1, 10, "uniform"),  # Scale weight for positive class
                # "subsample": Real(0.5, 1, "uniform"),  # Subsampling ratio for training
                # "tree_method": Categorical(["auto", "gpu_hist", "hist"]),  # Tree method for GPU (optional)
                # "validate_parameters": Categorical([None]),  # Validate parameters during training
            
        else:
            # Traditional Grid Search: Define parameter space using lists
            self.parameter_space = {
                "objective": ["binary:logistic"],  # Objective function for binary classification
                "booster": ["gbtree", "gblinear", "dart"],  # Type of boosting model
                "gamma": self.parameter_vector_space.param_dict.get("log_small"),  # Regularization parameter
                "grow_policy": ["depthwise", "lossguide"],  # Tree growth policy
                "learning_rate": self.parameter_vector_space.param_dict.get("log_small"),  # Learning rate
                "max_bin": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max bins for discretization
                "max_depth": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max depth of tree
                "max_leaves": self.parameter_vector_space.param_dict.get("log_large_long"),  # Max number of leaves
                "min_child_weight": [None],  # Minimum sum of instance weight in a child
                "n_estimators": self.parameter_vector_space.param_dict.get("log_large_long"),  # Number of boosting rounds
                "n_jobs": [-1],  # Number of parallel threads for training
                "random_state": [None],  # Random state for reproducibility
                "reg_alpha": self.parameter_vector_space.param_dict.get("log_small"),  # L1 regularization term
                "reg_lambda": self.parameter_vector_space.param_dict.get("log_small"),  # L2 regularization term
                "sampling_method": ["uniform"],  # Sampling method during training
                "verbosity": [0],  # Verbosity level during training

                # Future use parameters for Grid Search (currently commented out)
                # "use_label_encoder": [True, False],  # Use label encoder
                # "base_score": [0.0, 1.0],  # Base score for predictions
                # "callbacks": [None],  # Custom callbacks for training
                # "colsample_bylevel": [0.5, 1],  # Column sampling by level
                # "colsample_bynode": [0.5, 1],  # Column sampling by node
                # "colsample_bytree": [0.5, 1],  # Column sampling by tree
                # "early_stopping_rounds": [None],  # Early stopping for boosting rounds
                # "enable_categorical": [True, False],  # Enable categorical variables (if needed)
                # "eval_metric": [None],  # Evaluation metric (optional)
                # "gpu_id": [None],  # GPU id to use for training
                # "importance_type": ["weight", "gain", "cover"],  # Type of feature importance calculation
                # "interaction_constraints": [None],  # Constraints for feature interaction
                # "max_cat_to_onehot": [1, 100],  # Max categories for one-hot encoding
                # "max_delta_step": [0, 10],  # Max delta step for optimization
                # "monotone_constraints": [None],  # Constraints for monotonicity in predictions
                # "num_parallel_tree": [1, 10],  # Number of parallel trees in boosting
                # "predictor": ["cpu_predictor", "gpu_predictor"],  # Type of predictor (e.g., 'gpu_predictor')
                # "scale_pos_weight": [1, 10],  # Scale weight for positive class
                # "subsample": [0.5, 1],  # Subsampling ratio for training
                # "tree_method": ["auto", "gpu_hist", "hist"],  # Tree method for GPU (optional)
                # "validate_parameters": [None],  # Validate parameters during training
            }

        return None
