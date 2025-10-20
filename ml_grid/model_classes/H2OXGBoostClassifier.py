from h2o.estimators import H2OXGBoostEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd

class H2OXGBoostClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's XGBoost.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OXGBoostClassifier.

        All keyword arguments are passed directly to the H2OXGBoostEstimator.
        Example args: ntrees=50, max_depth=5
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2OXGBoostEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OXGBoostClassifier":
        """Fits the H2O XGBoost model with specific robustness checks."""
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- FIX for incorrect hyperparameter name ---
        # Map 'col_sample_rate_bytree' (from sklearn XGBoost) to 'col_sample_rate_per_tree' (H2O XGBoost)
        if 'col_sample_rate_bytree' in model_params:
            model_params['col_sample_rate_per_tree'] = model_params.pop('col_sample_rate_bytree')


        # --- CRITICAL FIX for server crashes on single constant feature ---
        # H2O's XGBoost can crash the server in this specific edge case.
        if len(x_vars) == 1 and X[x_vars[0]].nunique() <= 1:
            raise ValueError(
                "Dataset has a single constant feature. H2O XGBoost is unstable "
                "in this scenario. Halting execution."
            )

        # Train with the original parameters if no specific fallback was triggered
        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self