from h2o.estimators import H2ONaiveBayesEstimator
from .H2OBaseClassifier import H2OBaseClassifier

class H2ONaiveBayesClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Naive Bayes Classifier.
    """
    def __init__(self, **kwargs):
        """Initializes the H2ONaiveBayesClassifier.

        All keyword arguments are passed directly to the H2ONaiveBayesEstimator.
        Example args: laplace=1
        """
        super().__init__(H2ONaiveBayesEstimator, **kwargs)