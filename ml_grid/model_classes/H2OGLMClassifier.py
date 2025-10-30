
import pandas as pd
from h2o.estimators import H2OGeneralizedLinearEstimator

from .H2OBaseClassifier import H2OBaseClassifier


class H2OGLMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Generalized Linear Models."""

    def __init__(self, **kwargs):
        """Initializes the H2OGLMClassifier, handling the 'lambda' parameter.

        This wrapper ensures compatibility with scikit-learn's parameter naming
        by accepting `lambda_` and internally mapping it for the H2O backend.

        Args:
            **kwargs: Keyword arguments passed directly to the
                `H2OGeneralizedLinearEstimator`. Common arguments include
                `family='binomial'`, `alpha=0.5`, and `lambda_` (for regularization).
        """
        # --- FIX for scikit-learn cloning and H2O's 'lambda' parameter ---
        # scikit-learn's get_params() will return 'lambda_', but the user might
        # provide 'lambda' in the parameter grid. We must handle both cases.
        if "lambda" in kwargs and "lambda_" not in kwargs:
            kwargs["lambda_"] = kwargs.pop("lambda")

        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop("estimator_class", None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2OGeneralizedLinearEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OGLMClassifier":
        """Fits the H2O GLM model and corrects the 'lambda_' parameter name.

        This method first calls the parent `fit` method to train the model.
        After fitting, it ensures the internal H2O model object has the
        regularization parameter named 'lambda' (not 'lambda_') to prevent
        errors during prediction.

        Returns:
            H2OGLMClassifier: The fitted classifier instance.
        """
        # Call the parent class's fit method to perform the actual training
        super().fit(X, y, **kwargs)

        # --- CRITICAL FIX for predict-time NullPointerException ---
        # The H2O backend's predict method requires the 'lambda' parameter, but the
        # Python object may hold it as 'lambda_'. We must ensure the final model
        # object has the correct 'lambda' parameter set in its internal params dict.
        if self.model_ and "lambda_" in self.model_.params:
            self.model_.params["lambda"] = self.model_.params.pop("lambda_")

        return self
