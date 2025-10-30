"""H2O Naive Bayes Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for H2O's H2ONaiveBayesEstimator.
"""

from h2o.estimators import H2ONaiveBayesEstimator
from .H2OBaseClassifier import H2OBaseClassifier


class H2ONaiveBayesClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Naive Bayes Classifier."""

    def __init__(self, **kwargs):
        """Initializes the H2ONaiveBayesClassifier.

        All keyword arguments are passed directly to the H2ONaiveBayesEstimator.
        Example args: laplace=1
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop("estimator_class", None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2ONaiveBayesEstimator, **kwargs)
