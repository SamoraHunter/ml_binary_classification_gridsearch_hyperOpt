from typing import Optional
import pandas as pd
from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from skopt.space import Categorical, Real, Integer
import logging

logging.getLogger('ml_grid').debug("Imported h2o_gam_classifier_class")

class H2O_GAM_class:
    """H2OGAMClassifier with support for Bayesian and grid search parameter spaces."""

    def __init__(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        parameter_space_size: Optional[str] = None,
    ):
        self.X = X
        self.y = y
        
        # GAM requires specifying which columns to apply splines to.
        # This is now handled dynamically in the parameter space.
        self.algorithm_implementation = H2OGAMClassifier()
        self.method_name = "H2OGAMClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        
        # Define the available columns for the hyperparameter search to choose from.
        gam_cols = list(X.columns) if X is not None else []
            
        # Conditionally define the parameter space based on the search method.
        is_bayes_search = global_parameters.bayessearch

        if is_bayes_search:
            # For Bayesian search, use skopt distribution objects.
            current_param_space = {
                "num_knots": Integer(5, 15),
                "bs": Categorical(['cs', 'tp']),
                "scale": Real(0.01, 1.0, "log-uniform"),
                "seed": Integer(1, 1000),
            }
            if gam_cols:
                current_param_space["gam_columns"] = Categorical(gam_cols)
            self.parameter_space = [current_param_space]
        else:
            # For Grid/Random search, use standard lists.
            current_param_space = {
                "num_knots": [5, 8, 10, 12, 15],
                "bs": ['cs', 'tp'],
                "scale": [0.01, 0.1, 0.5, 1.0],
                "seed": [1, 42, 123, 500, 1000],
            }
            if gam_cols:
                current_param_space["gam_columns"] = gam_cols
            self.parameter_space = [current_param_space]