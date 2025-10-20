from h2o.estimators import H2ORandomForestEstimator

from .H2OBaseClassifier import H2OBaseClassifier

class H2ODRFClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Distributed Random Forest.

    This class allows H2O's DRF to be used as a standard scikit-learn
    classifier, making it compatible with tools like GridSearchCV and
    BayesSearchCV.
    """
    def __init__(self, **kwargs):
        """Initializes the H2ODRFClassifier.

        All keyword arguments are passed directly to the H2ORandomForestEstimator.
        Example args: ntrees=50, max_depth=20, seed=1
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2ORandomForestEstimator, **kwargs)