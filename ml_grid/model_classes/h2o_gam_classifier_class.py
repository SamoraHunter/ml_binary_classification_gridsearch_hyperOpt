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
        # We will use all available features from the dataset as the default.
        gam_cols = list(X.columns) if X is not None else []
        self.algorithm_implementation = H2OGAMClassifier(gam_columns=gam_cols)
        self.method_name = "H2OGAMClassifier"
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        
        if not gam_cols:
            logging.warning("H2O_GAM_class initialized with no columns for 'gam_columns'. This may fail if not provided later.")
            
        if global_parameters.bayessearch:
            self.parameter_space = [
                {
                    "num_knots": Integer(5, 20),
                    "bs": Categorical([0, 1]), # Use cubic regression splines or thin plate regression splines
                    "scale": Real(0.001, 1.0, "log-uniform"),
                    "seed": Integer(1, 1000),
                }
            ]
        else:
            self.parameter_space = [
                {
                    "num_knots": [5, 10, 15],
                    "bs": [0, 1],
                    "scale": [0.01, 0.1, 1.0],
                    "seed": [1, 42],
                }
            ]