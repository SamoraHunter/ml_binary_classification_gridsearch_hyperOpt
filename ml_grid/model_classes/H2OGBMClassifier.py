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

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Overrides the base _prepare_fit to add GBM-specific parameter validation.
        """
        # Call the base class's _prepare_fit to get the initial setup
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        n_samples = len(train_h2o)
        max_allowed_min_rows = max(1.0, n_samples / 2.0) if n_samples > 0 else 1.0

        current_min_rows = model_params.get('min_rows', 10.0)

        if current_min_rows > max_allowed_min_rows:
            self.logger.warning(f"Adjusting 'min_rows' from {current_min_rows} to {max_allowed_min_rows} to prevent H2O error.")
            model_params['min_rows'] = max_allowed_min_rows

        return train_h2o, x_vars, outcome_var, model_params

    # The fit() method is now inherited from H2OBaseClassifier.