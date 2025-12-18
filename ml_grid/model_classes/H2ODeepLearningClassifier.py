import pandas as pd
from h2o.estimators import H2ODeepLearningEstimator
from skopt.space import Categorical, Integer, Real

from ml_grid.util.global_params import global_parameters

from .H2OBaseClassifier import H2OBaseClassifier

PARAM_SPACE_GRID = {
    "xsmall": {
        "epochs": [5],
        "hidden_config": ["small"],
        "activation": ["Rectifier"],
        "l1": [0],
        "l2": [0],
        "seed": [1],
    },
    "small": {
        "epochs": [5, 10],
        "hidden_config": ["small", "medium"],
        "activation": ["Rectifier", "Tanh"],
        "l1": [0, 1e-4],
        "l2": [0, 1e-4],
        "seed": [1, 42],
    },
    "medium": {
        "epochs": [10, 50, 100],
        "hidden_config": ["small", "medium", "large"],
        "activation": ["Rectifier", "Tanh", "Maxout"],
        "l1": [0, 1e-4, 1e-3],
        "l2": [0, 1e-4, 1e-3],
        "seed": [1, 42, 123],
    },
}

PARAM_SPACE_BAYES = {
    "xsmall": {
        "epochs": Integer(5, 10),
        "hidden_config": Categorical(["small"]),
        "activation": Categorical(["Rectifier"]),
        "l1": Real(1e-5, 1e-4, "log-uniform"),
        "l2": Real(1e-5, 1e-4, "log-uniform"),
        "seed": Integer(1, 100),
    },
    "small": {
        "epochs": Integer(5, 20),
        "hidden_config": Categorical(["small", "medium", "large"]),
        "activation": Categorical(["Rectifier", "Tanh"]),
        "l1": Real(1e-5, 1e-3, "log-uniform"),
        "l2": Real(1e-5, 1e-3, "log-uniform"),
        "seed": Integer(1, 1000),
    },
    "medium": {
        "epochs": Integer(10, 200),
        "hidden_config": Categorical(["small", "medium", "large"]),
        "activation": Categorical(["Rectifier", "Tanh", "Maxout"]),
        "l1": Real(1e-6, 1e-2, "log-uniform"),
        "l2": Real(1e-6, 1e-2, "log-uniform"),
        "seed": Integer(1, 2000),
    },
}


class H2ODeepLearningClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Deep Learning models.

    This class handles special logic for the 'hidden' layer configuration.
    """

    def __init__(
        self, hidden=None, hidden_config=None, parameter_space_size="small", **kwargs
    ):
        """Initializes the H2ODeepLearningClassifier.

        It allows specifying hidden layers either directly via 'hidden' or
        through a predefined configuration name 'hidden_config'.

        Args:
            hidden (list, optional): A list of integers specifying the number of
                neurons for each hidden layer. Defaults to None.
            hidden_config (str, optional): A string key ('small', 'medium', 'large')
                to select a predefined hidden layer architecture. Defaults to None.
            **kwargs: Additional keyword arguments passed to the H2ODeepLearningEstimator.
        """
        # Set these as instance attributes for scikit-learn compatibility
        self.hidden = hidden
        self.hidden_config = hidden_config
        self.parameter_space_size = parameter_space_size

        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop("estimator_class", None)

        # Add our specific parameters to kwargs to be handled by the base class
        kwargs["hidden"] = self.hidden
        kwargs["hidden_config"] = self.hidden_config

        if parameter_space_size not in PARAM_SPACE_GRID:
            raise ValueError(
                f"Invalid parameter_space_size: '{parameter_space_size}'. Must be one of {list(PARAM_SPACE_GRID.keys())}"
            )

        if global_parameters.bayessearch:
            # For Bayesian search, the parameter space is a single dictionary
            self.parameter_space = PARAM_SPACE_BAYES[parameter_space_size]
        else:
            # For Grid search, the parameter space is a list of dictionaries
            self.parameter_space = [PARAM_SPACE_GRID[parameter_space_size]]

        # Pass all parameters to the super constructor
        super().__init__(estimator_class=H2ODeepLearningEstimator, **kwargs)

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Overrides the base _prepare_fit to resolve the hidden layer configuration
        before the model is instantiated.
        """
        # Call the base class's _prepare_fit to get the initial setup
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        # --- Deep Learning Specific Logic ---
        # If 'hidden' is not explicitly provided, use 'hidden_config' to set it.
        # We modify the model_params dictionary, not self.hidden.
        if model_params.get("hidden") is None:
            config_name = model_params.get("hidden_config") or "medium"

            hidden_layer_configs = {
                "small": [10, 10],
                "medium": [50, 50],
                "large": [100, 100, 100],
            }

            resolved_hidden = hidden_layer_configs.get(config_name, [50, 50])
            model_params["hidden"] = resolved_hidden
            self.logger.debug(
                f"Resolved hidden layers from config '{config_name}' to {resolved_hidden}"
            )

        # Remove the wrapper-only 'hidden_config' parameter before training
        model_params.pop("hidden_config", None)

        return train_h2o, x_vars, outcome_var, model_params

    # The fit() method is now inherited from H2OBaseClassifier and will use the
    # parameters returned by our overridden _prepare_fit().
