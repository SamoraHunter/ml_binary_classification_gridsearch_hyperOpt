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
        super().__init__(H2OGeneralizedLinearEstimator, **kwargs)