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
        super().__init__(H2ORuleFitEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2ORuleFitClassifier":
        """Fits the H2O RuleFit model with specific robustness checks."""
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # --- CRITICAL FIX for server crashes on single constant feature ---
        # H2O's RuleFit fails if the only feature is constant.
        if len(x_vars) == 1 and X[x_vars[0]].nunique() <= 1:
            print(
                f"Warning: Dataset has a single constant feature. "
                f"H2O RuleFit is unstable in this scenario. "
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