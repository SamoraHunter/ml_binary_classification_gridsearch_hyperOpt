import numpy as np
import pandas as pd
from h2o.estimators import H2OGeneralizedLinearEstimator

# Removing skopt imports to prevent the ParameterGrid TypeError
# from skopt.space import Real, Categorical, Integer

from .H2OBaseClassifier import H2OBaseClassifier


class H2OGLMClassifier(H2OBaseClassifier):
    """
    The actual scikit-learn compatible wrapper for H2O's Generalized Linear Models.
    This class handles the training, prediction, and H2O interaction.
    """

    def __init__(self, **kwargs):
        """Initializes the H2OGLMClassifier."""

        # --- FIX 1: Normalize lambda parameter name ---
        if "lambda" in kwargs and "lambda_" not in kwargs:
            kwargs["lambda_"] = kwargs.pop("lambda")

        kwargs.pop("estimator_class", None)

        # --- DEFENSIVE DEFAULTS ---
        kwargs.setdefault("standardize", True)

        # --- CRITICAL FIXES FOR STABILITY ---
        # We set these here for clarity, but the real enforcement happens in _prepare_fit
        kwargs["solver"] = "L_BFGS"
        kwargs["remove_collinear_columns"] = False
        kwargs["lambda_search"] = False

        # Pass the specific estimator class
        super().__init__(estimator_class=H2OGeneralizedLinearEstimator, **kwargs)

    def _prepare_fit(self, X, y):
        """
        Intercepts the parameter preparation to ENFORCE stability settings.
        This runs immediately BEFORE the H2O model is initialized/trained.
        """
        # Get the standard parameters from the base class
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        # --- STRICT OVERRIDE (The "Triple-Lock") ---
        # Regardless of what GridSearch/HyperOpt requested, we force these values
        # to prevent the Java Backend Crash (NullPointerException).

        # 1. Force L_BFGS: The only solver robust against the index mismatch bug on this data
        model_params["solver"] = "L_BFGS"

        # 2. Disable Collinear Removal: This prevents the coefficient vector size change
        model_params["remove_collinear_columns"] = False

        # 3. Disable Lambda Search: If True, H2O ignores 'solver' and uses Coordinate Descent
        model_params["lambda_search"] = False

        self.logger.info(
            f"H2OGLMClassifier: Enforced stability params: solver={model_params['solver']}, lambda_search={model_params['lambda_search']}"
        )

        return train_h2o, x_vars, outcome_var, model_params

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OGLMClassifier":
        """Fits the H2O GLM model."""
        # The override logic is now handled in _prepare_fit, called by super().fit()
        super().fit(X, y, **kwargs)
        return self


class H2O_GLM_class:
    """
    The Model Definition class used by the Grid Search framework.
    """

    def __init__(self, X=None, y=None, parameter_space_size="small"):
        self.method_name = "H2OGLMClassifier"

        # Instantiate the actual estimator wrapper
        self.algorithm_implementation = H2OGLMClassifier()

        # Define the Hyperparameter Space
        # FIX: Converted skopt distributions (Real, Categorical) to Lists
        # to ensure compatibility with sklearn.model_selection.ParameterGrid

        if parameter_space_size == "xsmall":
            self.parameter_space = {
                "alpha": [0.0, 0.5, 1.0],
                "lambda_": [1e-3, 1e-2, 1e-1],
                "family": ["binomial"],
                "solver": ["L_BFGS"],
                "standardize": [True],
            }
        elif parameter_space_size == "small":
            self.parameter_space = {
                "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
                "lambda_": np.logspace(-4, -1, 5).tolist(),
                "family": ["binomial"],
                "solver": ["L_BFGS"],
                "standardize": [True],
            }
        else:
            # Medium/Large space
            self.parameter_space = {
                "alpha": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "lambda_": np.logspace(-6, 1, 8).tolist(),
                "family": ["binomial"],
                "solver": ["L_BFGS"],
                "standardize": [True, False],
                "balance_classes": [True, False],
            }
