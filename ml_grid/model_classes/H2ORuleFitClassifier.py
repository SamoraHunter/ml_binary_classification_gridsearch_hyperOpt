from h2o.estimators import H2ORuleFitEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd

class H2ORuleFitClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's RuleFit.
    """
    def __init__(self, **kwargs):
        """Initializes the H2ORuleFitClassifier.

        All keyword arguments are passed directly to the H2ORuleFitEstimator.
        Example args: max_rule_length=3, model_type='rules_and_linear'
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2ORuleFitEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2ORuleFitClassifier":
        """Fits the H2O RuleFit model with specific robustness checks."""
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- CRITICAL FIX for server crashes on single constant feature ---
        # H2O's RuleFit fails if the only feature is constant.
        if len(x_vars) == 1 and X[x_vars[0]].nunique() <= 1:
            raise ValueError(
                "Dataset has a single constant feature. H2O RuleFit is unstable "
                "in this scenario. Halting execution."
            )

        # --- FIX for invalid parameter combinations ---
        # Before training, validate and correct dependent parameters to prevent server errors.
        min_len = model_params.get('min_rule_length')
        max_len = model_params.get('max_rule_length')

        if min_len is not None and max_len is not None and min_len > max_len:
            print(
                f"Warning: Invalid H2ORuleFit params detected: min_rule_length ({min_len}) > max_rule_length ({max_len}). "
                f"Correcting min_rule_length to {max_len} to proceed."
            )
            # Correct the invalid parameter by setting min_rule_length to max_rule_length.
            # This allows the hyperparameter search to continue with a valid configuration.
            model_params['min_rule_length'] = max_len

        # Train with the original parameters if no specific fallback was triggered
        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self