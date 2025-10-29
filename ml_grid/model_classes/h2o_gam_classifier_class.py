"""Configuration class for the H2O Generalized Additive Model (GAM) Classifier."""

from typing import Any, Dict, List, Optional

import logging
import pandas as pd
from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer, Real

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_gam_classifier_class")


class H2OGAMConfig:
    """Configuration class for H2OGAMClassifier.

    Provides parameter spaces for grid search and Bayesian optimization.
    The parameter space is dynamically generated to include columns from the
    input data `X` for the `gam_columns` parameter.

    Attributes:
        X (Optional[pd.DataFrame]): The input features.
        y (Optional[pd.Series]): The target variable.
        algorithm_implementation (H2OGAMClassifier): An instance of the
            classifier.
        method_name (str): The name of the method, "H2OGAMClassifier".
        parameter_space (List[Dict[str, Any]]): The hyperparameter search space.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> None:
        """Initializes the H2OGAMConfig.

        Args:
            X (Optional[pd.DataFrame]): The input features. This is used to
                dynamically populate the `gam_columns` in the parameter space.
            y (Optional[pd.Series]): The target variable.
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y
        self.algorithm_implementation = H2OGAMClassifier()
        self.method_name: str = "H2OGAMClassifier"

        # Define the available columns for the hyperparameter search.
        gam_cols = list(X.columns) if X is not None else []

        # Conditionally define the parameter space based on the search method.
        if global_parameters.bayessearch:
            # For Bayesian search, use skopt distribution objects.
            param_space = {
                "num_knots": Integer(5, 15),
                "bs": Categorical(["cs", "tp"]),
                "scale": Real(0.01, 1.0, "log-uniform"),
                "seed": Integer(1, 1000),
            }
            if gam_cols:
                # H2O GAM can take a list of lists for gam_columns, but for
                # hyperparameter search, we let it pick one column to focus on.
                # This could be extended to search for combinations.
                param_space["gam_columns"] = Categorical(gam_cols)
        else:
            # For Grid/Random search, use standard lists.
            param_space = {
                "num_knots": [5, 8, 10, 12, 15],
                "bs": ["cs", "tp"],
                "scale": [0.01, 0.1, 0.5, 1.0],
                "seed": [1, 42, 123, 500, 1000],
            }
            if gam_cols:
                param_space["gam_columns"] = gam_cols

        self.parameter_space: List[Dict[str, Any]] = [param_space]
