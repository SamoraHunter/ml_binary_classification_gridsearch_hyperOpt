import pandas as pd
from h2o.estimators import H2OGeneralizedLinearEstimator
from skopt.space import Real, Categorical

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

    def _prepare_fit(self, X, y):
        """
        Intercepts the parameter preparation to ENFORCE stability settings.
        This runs immediately BEFORE the H2O model is initialized/trained.
        """
        # Get the standard parameters from the base class
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        # --- STRICT OVERRIDE ---
        # Force L_BFGS: The only solver robust against the index mismatch bug on this data
        model_params["solver"] = "L_BFGS"
        model_params["remove_collinear_columns"] = False
        model_params["lambda_search"] = False

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
