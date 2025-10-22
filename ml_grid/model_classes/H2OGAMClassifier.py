import logging
import inspect # Make sure inspect is imported
from h2o.estimators import H2OGeneralizedAdditiveEstimator, H2OGeneralizedLinearEstimator
from .H2OBaseClassifier import H2OBaseClassifier
import pandas as pd
import numpy as np
import h2o
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__) # Use module-level logger for consistency

class H2OGAMClassifier(H2OBaseClassifier):
    """A scikit-learn compatible wrapper for H2O's Generalized Additive Models."""
    def __init__(self, **kwargs):
        """Initializes the H2OGAMClassifier.

        All keyword arguments are passed directly to the H2OGeneralizedAdditiveEstimator.
        Example args: family='binomial', gam_columns=['feature1']
        """
        kwargs.pop('estimator_class', None)
        super().__init__(estimator_class=H2OGeneralizedAdditiveEstimator, **kwargs)
        # self._using_dummy_model is handled by base class
        self._fallback_to_glm = False

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Overrides the base _prepare_fit to handle GAM-specific logic and fallback.
        """
        # Call the base class's _prepare_fit to get the initial setup
        train_h2o, x_vars, outcome_var, initial_model_params = super()._prepare_fit(X, y)

        model_params = initial_model_params.copy()
        self._fallback_to_glm = False # Reset flag

        # --- 1. Parameter Preprocessing for GAM ---
        if 'gam_columns' not in model_params or not model_params['gam_columns']:
             self.logger.warning("H2OGAMClassifier: 'gam_columns' not provided or empty. Defaulting to all numerical features.")
             numeric_cols = [col for col in x_vars if train_h2o[col].types[col] in ['int', 'real']]
             model_params['gam_columns'] = numeric_cols if numeric_cols else []
        elif isinstance(model_params['gam_columns'], tuple):
             model_params['gam_columns'] = list(model_params['gam_columns'])
        elif isinstance(model_params['gam_columns'], list) and model_params['gam_columns'] and isinstance(model_params['gam_columns'][0], list):
             model_params['gam_columns'] = [item for sublist in model_params['gam_columns'] for item in sublist]

        gam_columns = model_params.get('gam_columns', [])

        if 'bs' in model_params and gam_columns:
            bs_val = model_params['bs']
            bs_map = {'cs': 0, 'tp': 1}
            try:
                if isinstance(bs_val, str):
                    model_params['bs'] = [bs_map.get(bs_val, 0)] * len(gam_columns)
                elif isinstance(bs_val, list) and all(isinstance(b, str) for b in bs_val):
                    model_params['bs'] = [bs_map.get(b, 0) for b in bs_val]
            except Exception as e:
                self.logger.warning(f"Could not process 'bs' parameter: {e}. Using default.")
                model_params.pop('bs', None)

        num_knots_val = model_params.get('num_knots')
        if isinstance(num_knots_val, int) and gam_columns:
            model_params['num_knots'] = [num_knots_val] * len(gam_columns)

        scale_val = model_params.get('scale')
        if isinstance(scale_val, (int, float)) and gam_columns:
            model_params['scale'] = [scale_val] * len(gam_columns)

        # --- 2. Check GAM Column Suitability & Fallback Logic ---
        needs_fallback = False
        if gam_columns:
            suitable_gam_cols, suitable_knots, suitable_bs, suitable_scale = [], [], [], []
            num_knots_list = model_params.get('num_knots', [])
            bs_list = model_params.get('bs', [])
            scale_list = model_params.get('scale', [])

            if not isinstance(num_knots_list, list) or len(num_knots_list) != len(gam_columns):
                 default_knots = 5
                 self.logger.warning(f"num_knots list invalid or missing. Defaulting to {default_knots} knots for all {len(gam_columns)} GAM columns.")
                 num_knots_list = [default_knots] * len(gam_columns)
                 model_params['num_knots'] = num_knots_list

            for i, col in enumerate(gam_columns):
                if col not in X.columns:
                    self.logger.warning(f"GAM column '{col}' not found in input data X. Skipping.")
                    continue

                n_unique = X[col].nunique()
                required_knots = num_knots_list[i]

                if n_unique >= max(3, required_knots + 2):
                    suitable_gam_cols.append(col)
                    suitable_knots.append(required_knots)
                    if i < len(bs_list): suitable_bs.append(bs_list[i])
                    if i < len(scale_list): suitable_scale.append(scale_list[i])
                else:
                    self.logger.warning(
                        f"Excluding GAM column '{col}': {n_unique} unique values "
                        f"insufficient for {required_knots} knots (require >= {max(3, required_knots + 2)})."
                    )

            if not suitable_gam_cols:
                self.logger.warning("No suitable GAM columns found after checking cardinality. Falling back to GLM.")
                needs_fallback = True
            else:
                model_params['gam_columns'] = suitable_gam_cols
                model_params['num_knots'] = suitable_knots
                model_params['bs'] = suitable_bs if suitable_bs else model_params.pop('bs', None)
                model_params['scale'] = suitable_scale if suitable_scale else model_params.pop('scale', None)
        elif 'num_knots' in model_params or 'scale' in model_params or 'bs' in model_params:
             self.logger.warning("GAM-specific parameters provided but no valid gam_columns. Falling back to GLM.")
             needs_fallback = True

        # --- 3. Apply Fallback if Needed ---
        if needs_fallback:
            self.logger.warning("Setting up fallback to H2OGeneralizedLinearEstimator.")
            self._fallback_to_glm = True
            glm_param_keys = set(inspect.signature(H2OGeneralizedLinearEstimator).parameters.keys())
            model_params = {k: v for k, v in initial_model_params.items() if k in glm_param_keys}

        return train_h2o, x_vars, outcome_var, model_params

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OGAMClassifier":
        """Fits the H2O GAM model, falling back to GLM if necessary."""
        # The base class fit will call our overridden _prepare_fit
        original_estimator_class = self.estimator_class
        if self._fallback_to_glm:
            self.estimator_class = H2OGeneralizedLinearEstimator

        try:
            # Call the base class fit method, which will do all the heavy lifting
            super().fit(X, y, **kwargs)
        finally:
            # CRITICAL: Always restore the original estimator class
            self.estimator_class = original_estimator_class
            self.logger.debug(f"Restored self.estimator_class to {self.estimator_class.__name__}")

        return self

    def shutdown(self):
        """Shuts down the H2O cluster using the base class's safe logic."""
        super().shutdown()
    
    def _validate_min_samples_for_fit(self, X: pd.DataFrame, y: pd.Series) -> bool: # Renaming the method
        """Checks for small data and fits a dummy model if needed.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            True if fallback was used, False otherwise
        """
        # GAM doesn't have a hard limit, rely on base class handling
        return False

    # (Optional: Add predict/predict_proba overrides if GAM needs special handling,
    # otherwise they are inherited from H2OBaseClassifier)