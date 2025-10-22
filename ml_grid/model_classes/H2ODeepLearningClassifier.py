from h2o.estimators import H2ODeepLearningEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd
import logging

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

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Overrides the base _prepare_fit to resolve the hidden layer configuration
        before the model is instantiated.
        """
        # Call the base class's _prepare_fit to get the initial setup
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        # --- Deep Learning Specific Logic ---
        # If 'hidden' is not explicitly provided, use 'hidden_config' to set it.
        # We modify the model_params dictionary, not self.hidden.
        if model_params.get('hidden') is None:
            config_name = model_params.get('hidden_config') or 'medium'
            
            hidden_layer_configs = {
                'small': [10, 10],
                'medium': [50, 50],
                'large': [100, 100, 100]
            }
            
            resolved_hidden = hidden_layer_configs.get(config_name, [50, 50])
            model_params['hidden'] = resolved_hidden
            self.logger.debug(f"Resolved hidden layers from config '{config_name}' to {resolved_hidden}")

        # Remove the wrapper-only 'hidden_config' parameter before training
        model_params.pop('hidden_config', None)
            
        return train_h2o, x_vars, outcome_var, model_params

    # The fit() method is now inherited from H2OBaseClassifier and will use the
    # parameters returned by our overridden _prepare_fit().