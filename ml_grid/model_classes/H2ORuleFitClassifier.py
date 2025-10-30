"""H2O RuleFit Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for H2O's RuleFitEstimator.
"""

from h2o.estimators import H2ORuleFitEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd

class H2ORuleFitClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's RuleFit.
    """
    def __init__(self, **kwargs):
        """Initializes the H2ORuleFitClassifier.

        All keyword arguments are passed directly to the H2ORuleFitEstimator.
        Example args: max_rule_length=3, model_type='rules_and_linear'
        """
        # Remove estimator_class from kwargs if present (happens during sklearn clone)
        kwargs.pop('estimator_class', None)
        # Pass the specific estimator class
        super().__init__(estimator_class=H2ORuleFitEstimator, **kwargs)

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Overrides the base _prepare_fit to add RuleFit-specific validation.
        """
        # Call the base class's _prepare_fit to get the initial setup
        train_h2o, x_vars, outcome_var, model_params = super()._prepare_fit(X, y)

        # --- CRITICAL FIX for server crashes on single constant feature ---
        # H2O's RuleFit fails if the only feature is constant.
        if len(x_vars) == 1 and X[x_vars[0]].nunique() <= 1:
            raise ValueError(
                "H2ORuleFitClassifier: Dataset has a single constant feature, which is "
                "unstable for RuleFit. Halting execution."
            )

        # --- FIX for invalid parameter combinations ---
        min_len = model_params.get('min_rule_length')
        max_len = model_params.get('max_rule_length')

        if min_len is not None and max_len is not None and min_len > max_len:
            self.logger.warning(
                f"Warning: Invalid H2ORuleFit params detected: min_rule_length ({min_len}) > max_rule_length ({max_len}). "
                f"Correcting min_rule_length to {max_len} to proceed."
            )
            model_params['min_rule_length'] = max_len

        return train_h2o, x_vars, outcome_var, model_params

    # The fit() method is now inherited from H2OBaseClassifier and will use the
    # parameters returned by our overridden _prepare_fit().