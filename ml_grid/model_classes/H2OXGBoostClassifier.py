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