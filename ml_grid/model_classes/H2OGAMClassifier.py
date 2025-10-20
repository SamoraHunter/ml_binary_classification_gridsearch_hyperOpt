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
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2OGeneralizedAdditiveEstimator, **kwargs)
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
        gam_columns = model_params.get('gam_columns', []) # Default to empty list
        num_knots_val = model_params.get('num_knots') # Get value, could be int or None

        # --- FIX for H2OTypeError: num_knots must be a list ---
        # Hyperparameter search often provides a single integer for num_knots.
        # H2O's API requires a list of knots, one for each GAM column.
        # This code expands the single integer into a list of the correct length.
        if isinstance(num_knots_val, int) and gam_columns:
            num_knots_list = [num_knots_val] * len(gam_columns)
            model_params['num_knots'] = num_knots_list
            self.logger.debug(f"Expanded num_knots from {num_knots_val} to {num_knots_list}")

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
            raise ValueError(
                "A GAM column has cardinality < 3, which is incompatible with "
                "the num_knots requirement for 'cs' splines. Halting execution."
            )

        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        super().shutdown()