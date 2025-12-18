import numpy as np
import pandas as pd
from h2o.estimators import H2OGeneralizedLinearEstimator
from skopt.space import Real, Categorical, Integer

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
        # 1. Force L_BFGS: The only solver robust against the Java NPE on this data.
        kwargs["solver"] = "L_BFGS"
        # 2. Disable collinear removal: Changing vector size causes index mismatch crashes.
        kwargs["remove_collinear_columns"] = False
        # 3. Disable lambda_search: If True, H2O ignores 'solver' and uses Coordinate Descent, causing crashes.
        kwargs["lambda_search"] = False

        # Pass the specific estimator class
        super().__init__(estimator_class=H2OGeneralizedLinearEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OGLMClassifier":
        """Fits the H2O GLM model."""

        # --- DOUBLE-LOCK: Enforce stable parameters at fit time ---
        # GridSearch calls set_params() which might overwrite our safe defaults.
        # We explicitly revert them here before training.

        kwargs["solver"] = "L_BFGS"
        kwargs["remove_collinear_columns"] = False
        kwargs["lambda_search"] = False

        # Update internal H2O parameter dictionary if it exists
        if hasattr(self, "_parms"):
            self._parms["solver"] = "L_BFGS"
            self._parms["remove_collinear_columns"] = False
            self._parms["lambda_search"] = False

        # Proceed with standard fit
        super().fit(X, y, **kwargs)

        # 3. TRIPLE-LOCK: Ensure the internal model object respects this
        if hasattr(self, "model_") and self.model_ is not None:
            self.model_._parms["solver"] = "L_BFGS"
            self.model_._parms["remove_collinear_columns"] = False
            self.model_._parms["lambda_search"] = False

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
        # CRITICAL: We only offer L_BFGS to the optimizer.

        if parameter_space_size == "xsmall":
            self.parameter_space = {
                "alpha": Real(0.0, 1.0),
                "lambda_": Real(1e-3, 1e-1, prior="log-uniform"),
                "family": Categorical(["binomial"]),
                "solver": Categorical(["L_BFGS"]),
                "standardize": Categorical([True]),
            }
        elif parameter_space_size == "small":
            self.parameter_space = {
                "alpha": Real(0.0, 1.0),
                "lambda_": Real(1e-4, 1e-1, prior="log-uniform"),
                "family": Categorical(["binomial"]),
                "solver": Categorical(["L_BFGS"]),
                "standardize": Categorical([True]),
            }
        else:
            # Medium/Large space
            self.parameter_space = {
                "alpha": Real(0.0, 1.0),
                "lambda_": Real(1e-6, 10.0, prior="log-uniform"),
                "family": Categorical(["binomial"]),
                "solver": Categorical(["L_BFGS"]),
                "standardize": Categorical([True, False]),
                "balance_classes": Categorical([True, False]),
            }
