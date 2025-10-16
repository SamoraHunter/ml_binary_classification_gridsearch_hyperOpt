from h2o.estimators import H2ODeepLearningEstimator
from .H2OBaseClassifier import H2OBaseClassifier

class H2ODeepLearningClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Deep Learning.
    """
    def __init__(self, **kwargs):
        """Initializes the H2ODeepLearningClassifier.

        All keyword arguments are passed directly to the H2ODeepLearningEstimator.
        Example args: epochs=10, hidden=[10, 10]
        """
        super().__init__(H2ODeepLearningEstimator, **kwargs)