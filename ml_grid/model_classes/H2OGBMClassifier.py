from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd
from skopt.space import Real, Integer
from ml_grid.util.global_params import global_parameters

from .H2OBaseClassifier import H2OBaseClassifier

# Define parameter spaces outside the class for better organization and reusability.
PARAM_SPACE_GRID = {
    "xsmall": {
        "ntrees": [50],
        "max_depth": [5],
        "learn_rate": [0.1],
        "sample_rate": [0.8],
        "col_sample_rate": [0.8],
        "seed": [1],
    },
    "small": {
        "ntrees": [50, 100, 200],
        "max_depth": [3, 5, 10],
        "learn_rate": [0.01, 0.1],
        "sample_rate": [0.8, 1.0],
        "col_sample_rate": [0.8, 1.0],
        "seed": [1, 42],
    },
    "medium": {
        "ntrees": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 15],
        "learn_rate": [0.01, 0.05, 0.1],
        "sample_rate": [0.7, 0.8, 0.9, 1.0],
        "col_sample_rate": [0.7, 0.8, 0.9, 1.0],
        "seed": [1, 42, 123],
    }
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "ntrees": Integer(50, 100),
        "max_depth": Integer(3, 5),
        "learn_rate": Real(0.05, 0.15, "log-uniform"),
        "sample_rate": Real(0.7, 0.9),
        "col_sample_rate": Real(0.7, 0.9),
        "seed": Integer(1, 100),
    },
    "small": {
        "ntrees": Integer(50, 500),
        "max_depth": Integer(3, 10),
        "learn_rate": Real(0.01, 0.2, "log-uniform"),
        "sample_rate": Real(0.5, 1.0),
        "col_sample_rate": Real(0.5, 1.0),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "ntrees": Integer(50, 1000),
        "max_depth": Integer(3, 20),
        "learn_rate": Real(0.005, 0.2, "log-uniform"),
        "sample_rate": Real(0.5, 1.0),
        "col_sample_rate": Real(0.5, 1.0),
        "seed": Integer(1, 2000),
    }
}

class H2OGBMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Gradient Boosting Machine.

    This class allows H2O's GBM to be used as a standard scikit-learn
    classifier, making it compatible with tools like GridSearchCV and
    BayesSearchCV.
    """
    def __init__(self, parameter_space_size='small', **kwargs):
        """Initializes the H2OGBMClassifier.

        All keyword arguments are passed directly to the H2OGradientBoostingEstimator.
        Example args: ntrees=50, max_depth=5, learn_rate=0.1, seed=1
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        
        self.parameter_space_size = parameter_space_size
        
        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}")

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]
            
        # Pass estimator_class as a keyword argument
        super().__init__(estimator_class=H2OGradientBoostingEstimator, **kwargs)

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Overrides the base _prepare_fit to add GBM-specific parameter validation.
        """
        # Call the base class's _prepare_fit to get the initial setup
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        n_samples = len(train_h2o)
        max_allowed_min_rows = max(1.0, n_samples / 2.0) if n_samples > 0 else 1.0

        current_min_rows = model_params.get('min_rows', 10.0)

        if current_min_rows > max_allowed_min_rows:
            self.logger.warning(f"Adjusting 'min_rows' from {current_min_rows} to {max_allowed_min_rows} to prevent H2O error.")
            model_params['min_rows'] = max_allowed_min_rows

        return train_h2o, x_vars, outcome_var, model_params

    # The fit() method is now inherited from H2OBaseClassifier.