from h2o.estimators import H2OGeneralizedLinearEstimator
from .H2OBaseClassifier import H2OBaseClassifier

class H2OGLMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Generalized Linear Models.
    """
    def __init__(self, **kwargs):
        """Initializes the H2OGLMClassifier.

        All keyword arguments are passed directly to the H2OGeneralizedLinearEstimator.
        Example args: family='binomial', alpha=0.5
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2OGeneralizedLinearEstimator, **kwargs)