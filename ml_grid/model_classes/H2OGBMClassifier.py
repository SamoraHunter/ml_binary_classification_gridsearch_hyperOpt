from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd

from .H2OBaseClassifier import H2OBaseClassifier

class H2OGBMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Gradient Boosting Machine.

    This class allows H2O's GBM to be used as a standard scikit-learn
    classifier, making it compatible with tools like GridSearchCV and
    BayesSearchCV.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OGBMClassifier.

        All keyword arguments are passed directly to the H2OGradientBoostingEstimator.
        Example args: ntrees=50, max_depth=5, learn_rate=0.1, seed=1
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass estimator_class as a keyword argument
        super().__init__(estimator_class=H2OGradientBoostingEstimator, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OGBMClassifier":
        """Fits the H2O GBM model with specific robustness checks."""
        if self._handle_small_data_fallback(X, y):
            return self

        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)
        
        n_samples = len(train_h2o)
        max_allowed_min_rows = max(1.0, n_samples / 2.0) if n_samples > 0 else 1.0

        current_min_rows = model_params.get('min_rows', 10.0)

        if current_min_rows > max_allowed_min_rows:
            model_params['min_rows'] = max_allowed_min_rows

        # Train with the (potentially adjusted) parameters if no fallback was triggered
        self.model = self.estimator_class(**model_params)
        self.model.train(y=outcome_var, x=x_vars, training_frame=train_h2o)
        return self