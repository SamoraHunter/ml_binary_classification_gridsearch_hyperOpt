"""H2O Generalized Additive Model (GAM) Classifier.

This module contains the H2OGAMClass, which is a configuration class for
the H2OGAMClassifier. It provides parameter spaces for grid search and
Bayesian optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Integer, Real

logger = logging.getLogger(__name__)
logger.debug("Imported h2o_gam_classifier_class")


class H2OGAMClass:
    """A configuration class for the H2OGAMClassifier.

    Provides parameter spaces for grid search and Bayesian optimization.
    The parameter space is dynamically generated to include columns from the
    input data `X` for the `gam_columns` parameter.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,  # type: ignore
        parameter_space_size: str = "small",  # Added for consistency
    ) -> None:
        """Initializes the H2OGAMClass.

        Args:
            X: The input features. This is used to
                dynamically populate the `gam_columns` in the parameter space.
            y: The target variable. # type: ignore
            parameter_space_size (str): The size of the parameter space to use
                ('xsmall', 'small', 'medium'). Defaults to 'small'.

        Raises:
            ValueError: If `parameter_space_size` is not a valid key (though current
                implementation does not explicitly raise this for GAM).
        """
        self.X: Optional[pd.DataFrame] = X
        self.y: Optional[pd.Series] = y
        self.algorithm_implementation: H2OGAMClassifier = H2OGAMClassifier()
        self.method_name: str = "H2OGAMClassifier"
        self.parameter_space: Union[List[Dict[str, Any]], Dict[str, Any]]

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
            self.parameter_space = param_space
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

            self.parameter_space = [param_space]
