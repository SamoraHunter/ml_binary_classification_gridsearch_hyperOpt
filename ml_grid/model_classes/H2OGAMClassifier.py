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
    def __init__(self, _suppress_low_cardinality_error=True, **kwargs):
        """Initializes the H2OGAMClassifier.

        All keyword arguments are passed directly to the H2OGeneralizedAdditiveEstimator.
        Example args: family='binomial', gam_columns=['feature1']
        
        Args:
            _suppress_low_cardinality_error (bool): If True, safely removes GAM columns with
                insufficient unique values. If False, raises a ValueError.
        """
        kwargs.pop('estimator_class', None)
        super().__init__(estimator_class=H2OGeneralizedAdditiveEstimator, **kwargs)
        self._suppress_low_cardinality_error = _suppress_low_cardinality_error
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
        self.logger.debug(f"DEBUG: Before GAM column processing, model_params['gam_columns'] type: {type(model_params.get('gam_columns'))}, value: {model_params.get('gam_columns')}")

        if 'gam_columns' not in model_params or not model_params['gam_columns']:
             self.logger.warning("H2OGAMClassifier: 'gam_columns' not provided or empty. Defaulting to all numerical features.")
             numeric_cols = [col for col in x_vars if train_h2o[col].types[col] in ['int', 'real']]
             model_params['gam_columns'] = numeric_cols if numeric_cols else []
        # --- FIX: Handle single string from BayesSearch ---
        elif isinstance(model_params['gam_columns'], str):
             model_params['gam_columns'] = [model_params['gam_columns']]
        elif isinstance(model_params['gam_columns'], tuple):
             model_params['gam_columns'] = list(model_params['gam_columns'])
        # --- FIX for TypeError: object of type 'int' has no len() ---
        elif isinstance(model_params['gam_columns'], int):
             # If an integer is passed (e.g., from a hyperparameter search),
             # convert it to a list containing the column name as a string.
             # H2O expects column names to be strings.
             model_params['gam_columns'] = [str(model_params['gam_columns'])]
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
        
        # --- ROBUSTNESS FIX: Ensure num_knots, bs, and scale are always lists matching gam_columns length ---
        num_knots_val = model_params.get('num_knots')
        if isinstance(num_knots_val, int) and gam_columns:
            model_params['num_knots'] = [num_knots_val] * len(gam_columns)

        scale_val = model_params.get('scale')
        if isinstance(scale_val, (int, float)) and gam_columns:
            model_params['scale'] = [scale_val] * len(gam_columns)
        
        bs_val = model_params.get('bs')
        if isinstance(bs_val, int) and gam_columns:
             model_params['bs'] = [bs_val] * len(gam_columns)
        # --- END FIX ---
        
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

                # H2O's backend requires num_knots < n_unique.
                if n_unique <= required_knots:
                    if not self._suppress_low_cardinality_error:
                        raise ValueError(
                            f"Number of knots ({required_knots}) must be at least one less than the number of unique values ({n_unique}) for feature '{col}'."
                        )
                    self.logger.warning(
                        f"Excluding GAM column '{col}': {n_unique} unique values "
                        f"insufficient for {required_knots} knots (require >= {required_knots + 1})."
                    )
                    continue

                # Pre-check for well-defined knots
                try:
                    quantiles = np.linspace(0, 1, required_knots)
                    knot_values = X[col].quantile(quantiles)
                    if knot_values.nunique() < required_knots:
                        self.logger.warning(
                            f"Excluding GAM column '{col}': Not enough unique values to generate distinct knots."
                        )
                        continue
                except Exception as e:
                    self.logger.warning(f"Excluding GAM column '{col}' due to an error during knot pre-check: {e}")
                    continue
                
                suitable_gam_cols.append(col)
                suitable_knots.append(required_knots)
                if i < len(bs_list): suitable_bs.append(bs_list[i])
                if scale_list and i < len(scale_list): suitable_scale.append(scale_list[i])

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
        # We need to explicitly call _prepare_fit here to set _fallback_to_glm
        # and get the processed parameters.
        
        # --- CRITICAL FIX: Manually call validation ---
        # This ensures that if X is a numpy array, it's converted to a DataFrame
        # with string columns before being passed to _prepare_fit.
        X, y = self._validate_input_data(X, y)
        # Call our overridden _prepare_fit to determine fallback and get processed data/params.
        # This method will internally call super()._prepare_fit which handles validation,
        # setting classes_, feature_names_, feature_types_, and H2OFrame creation.
        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

        # Determine the actual H2O estimator class to use
        if self._fallback_to_glm:
            self.logger.warning("H2OGAMClassifier.fit: Fallback to GLM triggered. Using H2OGeneralizedLinearEstimator.")
            h2o_estimator_to_use = H2OGeneralizedLinearEstimator
        else:
            h2o_estimator_to_use = self.estimator_class # This is H2OGeneralizedAdditiveEstimator

        # Instantiate the H2O model with all the hyperparameters
        self.logger.debug(f"Creating H2O model ({h2o_estimator_to_use.__name__}) with params: {model_params}")
        self.model_ = h2o_estimator_to_use(**model_params)
        
        # Call the train() method with ONLY the data-related arguments
        self.logger.debug("Calling H2O model.train()...")
        self.model_.train(x=x_vars, y=outcome_var, training_frame=train_h2o)

        # Store model_id for recovery - THIS IS CRITICAL for predict() to work
        self.logger.debug(f"H2O train complete, extracting model_id from {self.model_}")
        self.model_id = self.model_.model_id

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