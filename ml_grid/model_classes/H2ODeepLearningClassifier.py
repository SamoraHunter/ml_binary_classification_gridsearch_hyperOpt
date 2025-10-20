from h2o.estimators import H2ODeepLearningEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd

class H2ODeepLearningClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Deep Learning models.

    This class handles special logic for the 'hidden' layer configuration.
    """

    def __init__(self, hidden=None, hidden_config=None, **kwargs):
        """Initializes the H2ODeepLearningClassifier.

        It allows specifying hidden layers either directly via 'hidden' or
        through a predefined configuration name 'hidden_config'.

        Args:
            hidden (list, optional): A list of integers specifying the number of
                neurons for each hidden layer. Defaults to None.
            hidden_config (str, optional): A string key ('small', 'medium', 'large')
                to select a predefined hidden layer architecture. Defaults to None.
            **kwargs: Additional keyword arguments passed to the H2ODeepLearningEstimator.
        """
        # Set these as instance attributes for scikit-learn compatibility
        self.hidden = hidden
        self.hidden_config = hidden_config

        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)

        # Add our specific parameters to kwargs to be handled by the base class
        kwargs['hidden'] = self.hidden
        kwargs['hidden_config'] = self.hidden_config

        # Pass all parameters to the super constructor
        super().__init__(
            estimator_class=H2ODeepLearningEstimator, **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2ODeepLearningClassifier":
        """Fits the H2O Deep Learning model.

        It resolves the hidden layer configuration before calling the base fit method.
        """
        # Logic to resolve hidden layer configuration
        # If 'hidden' is not explicitly provided, use 'hidden_config'
        if self.hidden is None:
            config_name = self.hidden_config or 'medium'
            
            hidden_layer_configs = {
                'small': [10, 10],
                'medium': [50, 50],
                'large': [100, 100, 100]
            }

            # Set the 'hidden' parameter based on the config
            self.hidden = hidden_layer_configs.get(config_name, [50, 50])

        # Call the base class's fit method
        return super().fit(X, y)

    def _get_model_params(self):
        """Overrides the base method to handle special parameters.
        
        This removes the wrapper-specific 'hidden_config' before passing
        parameters to the underlying H2O estimator.
        """
        # Get params from the parent class's implementation
        params = super()._get_model_params()
        
        # Remove the wrapper-only parameter
        params.pop('hidden_config', None)
            
        return params
    