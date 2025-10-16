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
        super().__init__(H2OXGBoostEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OXGBoostClassifier":
        """Fits the H2O XGBoost model with specific robustness checks."""
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- CRITICAL FIX for server crashes on single constant feature ---
        # H2O's XGBoost can crash the server in this specific edge case.
        if len(x_vars) == 1 and X[x_vars[0]].nunique() <= 1:
            print(
                f"Warning: Dataset has a single constant feature. "
                f"H2O XGBoost is unstable in this scenario. "
                f"Skipping training and using a dummy model."
            )
            from h2o.estimators import H2OGeneralizedLinearEstimator
            self._using_dummy_model = True
            dummy_model = H2OGeneralizedLinearEstimator(family='binomial', ignore_const_cols=False)
            dummy_model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
            self.model = dummy_model
            return self

        # Train with the original parameters if no specific fallback was triggered
        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self