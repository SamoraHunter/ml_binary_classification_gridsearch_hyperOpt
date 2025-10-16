from h2o.estimators import H2OGeneralizedAdditiveEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd
import numpy as np
import h2o
from sklearn.utils.validation import check_is_fitted

class H2OGAMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Generalized Additive Models.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OGAMClassifier.

        All keyword arguments are passed directly to the H2OGeneralizedAdditiveEstimator.
        Example args: family='binomial', gam_columns=['feature1']
        """
        super().__init__(H2OGeneralizedAdditiveEstimator, **kwargs)
        self._using_dummy_model = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OGAMClassifier":
        """Fits the H2O GAM model with specific robustness checks for num_knots."""
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # GAM requires a list of columns for the splines. Check for it here.
        if not model_params.get('gam_columns'):
            raise ValueError("H2OGAMClassifier requires 'gam_columns' parameter in its fit parameters.")

        # --- CRITICAL FIX for num_knots vs cardinality ---
        # Ensure num_knots is not greater than the cardinality of the gam_columns.
        gam_columns = model_params.get('gam_columns', [])
        num_knots = model_params.get('num_knots', [])

        # --- CRITICAL FIX for num_knots vs cardinality ---
        # Check if any GAM column has a cardinality too low for the knot requirement.
        # H2O's cs splines require num_knots >= 3.
        is_unstable = False
        if gam_columns:
            for col in gam_columns:
                if X[col].nunique() < 3:
                    is_unstable = True
                    break
        
        if is_unstable:
            print(
                f"Warning: A GAM column has cardinality < 3, which is incompatible with "
                f"the num_knots requirement. Skipping GAM and using a dummy model."
            )
            from h2o.estimators import H2OGeneralizedLinearEstimator
            self._using_dummy_model = True
            dummy_model = H2OGeneralizedLinearEstimator(family='binomial', ignore_const_cols=False)
            dummy_model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            self.model = dummy_model
            return self

        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels, handling the dummy model edge case."""
        check_is_fitted(self)
        if self._using_dummy_model:
            return np.full(len(X), self.classes_[0])
        
        return super().predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities, handling the dummy model edge case."""
        check_is_fitted(self)
        if self._using_dummy_model:
            n_samples = len(X)
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))
            proba[:, 0] = 1.0
            return proba

        return super().predict_proba(X)

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        super().shutdown()