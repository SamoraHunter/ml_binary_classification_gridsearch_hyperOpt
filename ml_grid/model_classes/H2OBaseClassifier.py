from typing import Any, Dict, Optional
import inspect
import logging
import h2o
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from ml_grid.util.global_params import global_parameters

# Configure logging
logger = logging.getLogger(__name__)


class H2OBaseClassifier(BaseEstimator, ClassifierMixin):
    """A base class for scikit-learn compatible H2O classifier wrappers.

    This class provides common functionality for H2O model wrappers, including:
    - H2O cluster management (initialization and shutdown).
    - scikit-learn API compatibility (`get_params`, `set_params`).
    - Common `predict` and `predict_proba` implementations.
    - Robust handling of small datasets in the `fit` method.
    """

    MIN_SAMPLES_FOR_STABLE_FIT = 10

    def __init__(self, estimator_class=None, **kwargs):
        """Initializes the H2OBaseClassifier.

        Args:
            estimator_class: The H2O estimator class (e.g., H2OGradientBoostingEstimator).
                Can be passed as positional or keyword argument.
            **kwargs: Keyword arguments passed to the H2O estimator.
        """
        # Handle estimator_class - it might come in kwargs during cloning
        # or as a positional argument during normal instantiation
        if estimator_class is None and 'estimator_class' in kwargs:
            estimator_class = kwargs.pop('estimator_class')
        elif 'estimator_class' in kwargs:
            # If passed both ways, remove from kwargs to avoid conflict
            kwargs.pop('estimator_class')
        
        if estimator_class is None:
            raise ValueError("estimator_class is required")
            
        self.estimator_class = estimator_class
        
        # --- FIX: Ensure lambda is never stored as 'lambda', always as 'lambda_' ---
        # If lambda_ is in kwargs, keep it as lambda_
        # If lambda is in kwargs (shouldn't happen but be safe), convert to lambda_
        if 'lambda' in kwargs:
            kwargs['lambda_'] = kwargs.pop('lambda')
        
        # Set all kwargs as attributes for proper sklearn compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize logger for this instance
        # Use the project-wide logger for consistency
        self.logger = logging.getLogger('ml_grid')
        
        # Internal state attributes (not parameters)
        self.model: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list] = None
        self._is_cluster_owner = False
        self._was_fit_on_constant_feature = False
        self._using_dummy_model = False
        self._rename_cols_on_predict = True  # Default behavior
        # H2O models are not safe with joblib's process-based parallelism.
        self._n_jobs = 1

    def _ensure_h2o_is_running(self):
        """Safely checks for and initializes an H2O cluster if not running."""
        cluster = h2o.cluster()
        show_progress = getattr(global_parameters, 'h2o_show_progress', False)

        if not (cluster and cluster.is_running()):
            h2o.init()
            self._is_cluster_owner = True
        
        # Set progress bar visibility based on the global parameter
        h2o.no_progress() if not show_progress else h2o.show_progress()

    def _validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Validates and converts input data to proper format.
        
        Args:
            X: Feature matrix
            y: Target vector (optional, for fit-time validation)
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If data is invalid
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X = pd.DataFrame(X)
        
        # Reset index to avoid sklearn CV indexing issues
        if not isinstance(X.index, pd.RangeIndex):
            X = X.reset_index(drop=True)
        
        # Check for empty data
        if X.empty:
            raise ValueError("Cannot process empty DataFrame")
        
        # Validate y if provided (fit time)
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
            
            # Check for NaNs in the target variable
            # Handle different types: Series, numpy array, categorical
            if isinstance(y, pd.Series):
                has_nans = y.isnull().any()
            elif isinstance(y, pd.Categorical):
                # Categorical doesn't support np.isnan, use isnull on the codes
                has_nans = pd.isna(y.codes).any() or (y.codes == -1).any()
            else:
                # Assume numpy array or similar
                try:
                    has_nans = np.isnan(y).any()
                except (TypeError, ValueError):
                    # If np.isnan fails, try pandas isna
                    has_nans = pd.isna(y).any()
            
            if has_nans:
                raise ValueError(
                    "Target variable y contains NaN values, which is not supported."
                )

            # Get unique classes
            if isinstance(y, pd.Series):
                unique_classes = y.unique()
            elif isinstance(y, pd.Categorical):
                unique_classes = y.categories
            else:
                unique_classes = np.unique(y[~pd.isna(y)])
            
            if len(unique_classes) < 2:
                raise ValueError(f"y must have at least 2 classes, found {len(unique_classes)}")
        
        # Validate feature names match (predict time)
        if self.feature_names_ is not None and y is None:
            if list(X.columns) != self.feature_names_:
                # Check if we can reorder
                missing_features = set(self.feature_names_) - set(X.columns)
                if missing_features:
                    raise ValueError(
                        f"Missing required features: {missing_features}. "
                        f"Expected: {self.feature_names_}, got: {list(X.columns)}"
                    )
                # Reorder to match training
                X = X[self.feature_names_]
                logger.info("Reordered features to match training order")
        
        # Check for NaN values
        if X.isnull().any().any():
            raise ValueError(
                "Input data contains NaN values. Please handle missing values before fitting/predicting."
            )
        
        return X

    def _prepare_fit(self, X: pd.DataFrame, y: pd.Series):
        """Prepares data and parameters for fitting.
        
        Returns:
            Tuple of (train_h2o, x_vars, outcome_var, model_params)
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")

        self.classes_ = np.unique(y)
        outcome_var = y.name
        x_vars = list(X.columns)
        
        # Store feature names for validation during predict
        self.feature_names_ = x_vars

        self._ensure_h2o_is_running()

        # Convert target to pandas categorical type BEFORE creating H2OFrame
        # This ensures H2O recognizes it as categorical for binomial models
        y_categorical = pd.Categorical(y)
        y_series = pd.Series(y_categorical, name=outcome_var, index=y.index)
        
        train_df = pd.concat([X, y_series], axis=1)
        train_h2o = h2o.H2OFrame(train_df)
        
        # Explicitly convert the outcome column to factor
        # H2O should now recognize it as categorical since pandas sent it as categorical
        train_h2o[outcome_var] = train_h2o[outcome_var].asfactor()

        # Get model parameters from instance attributes
        model_params = self._get_model_params()

        # If there's only one feature, prevent H2O from dropping it if it's constant
        if len(x_vars) == 1 and self.estimator_class:
            if X[x_vars[0]].nunique() <= 1:
                self._was_fit_on_constant_feature = True
                logger.warning("Fitting on a single constant feature - predictions may be unreliable")

            estimator_params = inspect.signature(self.estimator_class).parameters
            if 'ignore_const_cols' in estimator_params:
                model_params.setdefault('ignore_const_cols', False)

        return train_h2o, x_vars, outcome_var, model_params

    def _get_model_params(self) -> Dict[str, Any]:
        """Extracts model parameters from instance attributes.
        
        Returns:
            Dictionary of parameters to pass to H2O estimator
        """
        # Get all possible parameters from the instance (excluding 'estimator_class')
        all_params = {k: v for k, v in self.get_params(deep=False).items() 
                      if k != 'estimator_class'}
        
        # Get the set of valid parameter names from the H2O estimator's constructor
        valid_param_keys = set(inspect.signature(self.estimator_class).parameters.keys())

        # Filter all_params to include only keys that are valid for the constructor
        model_params = {
            key: value
            for key, value in all_params.items()
            if key in valid_param_keys
        }

        return model_params

    def _handle_small_data_fallback(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Checks for small data and fits a dummy model if needed.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            True if fallback was used, False otherwise
        """
        if len(X) < self.MIN_SAMPLES_FOR_STABLE_FIT:
            raise ValueError(
                f"Dataset is too small ({len(X)} rows < {self.MIN_SAMPLES_FOR_STABLE_FIT}). "
                "H2O models are unstable on very small datasets. Halting execution."
            )
        return False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "H2OBaseClassifier":
        """Fits the H2O model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate input data
        X = self._validate_input_data(X, y)
        
        # Check if all features are constant, which can cause H2O to fail.
        if X.shape[1] > 0 and (X.nunique() == 1).all():
            raise ValueError("All features are constant. Halting execution as model fitting will fail.")

        # Check for small data
        if self._handle_small_data_fallback(X, y):
            return self

        # Fit the actual model
        train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)
        
        # Instantiate the H2O model with all the hyperparameters
        self.model = self.estimator_class(**model_params)
        # Call the train() method with ONLY the data-related arguments
        self.model.train(x=x_vars, y=outcome_var, training_frame=train_h2o)
        
        logger.debug(f"Successfully fitted {self.estimator_class.__name__}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted class labels
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If prediction fails
        """
        check_is_fitted(self)

        # If using dummy model due to constant features, return conservative prediction
        if self._was_fit_on_constant_feature:
            raise RuntimeError("Predicting on a model that was fit on a single constant feature is unreliable. Halting.")

        if self.model is None:
            raise RuntimeError("Model is None - this should not happen after fit()")

        # Validate input
        X = self._validate_input_data(X)
        
        # Ensure H2O is running
        self._ensure_h2o_is_running()
        
        # Create H2O frame with explicit column names to prevent schema mismatch
        try:
            test_h2o = h2o.H2OFrame(X, column_names=self.feature_names_)
        except Exception as e:
            raise RuntimeError(f"Failed to create H2O frame for prediction: {e}")
        
        # Make prediction
        try:
            predictions = self.model.predict(test_h2o)
        except Exception as e:
            raise RuntimeError(f"H2O prediction failed: {e}")
        
        # Extract predictions
        pred_df = predictions.as_data_frame(use_multi_thread=True)
        if "predict" in pred_df.columns:
            return pred_df["predict"].values.ravel()
        else:
            raise RuntimeError("Prediction output missing 'predict' column")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If prediction fails
        """
        check_is_fitted(self)

        # If using dummy model, return conservative probability distribution
        if self._was_fit_on_constant_feature:
            raise RuntimeError("Predicting probabilities on a model that was fit on a single constant feature is unreliable. Halting.")

        if self.model is None:
            raise RuntimeError("Model is None - this should not happen after fit()")

        # Validate input
        X = self._validate_input_data(X)
        
        # Ensure H2O is running
        self._ensure_h2o_is_running()
        
        # Create H2O frame with explicit column names to prevent schema mismatch
        try:
            test_h2o = h2o.H2OFrame(X, column_names=self.feature_names_)
        except Exception as e:
            raise RuntimeError(f"Failed to create H2O frame for prediction: {e}")
        
        # Make prediction
        try:
            predictions = self.model.predict(test_h2o)
        except Exception as e:
            raise RuntimeError(f"H2O prediction failed: {e}")
        
        # Extract probabilities (drop the 'predict' column)
        prob_df = predictions.drop("predict").as_data_frame(use_multi_thread=True)
        return prob_df.values

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Gets parameters for this estimator.

        This is the canonical scikit-learn implementation for an estimator
        that uses **kwargs in its __init__ method. It correctly identifies
        hyperparameters and separates them from fitted attributes.
        """
        params = {}
        
        # Add all public attributes that represent constructor parameters
        # Exclude: private attributes (_*), fitted attributes (*_), and internal state
        INTERNAL_ATTRS = {
            'model',   # H2O model instance (created during fit)
            'logger',  # Logger instance (not a parameter)
            'classes_',  # Fitted attribute (class labels)
            'feature_names_',  # Fitted attribute (feature names from training)
        }
        
        for key in self.__dict__:
            if key.startswith('_'):  # Skip private attributes
                continue
            if key in INTERNAL_ATTRS:  # Skip known internal state
                continue
            # Special handling: attributes ending in _ are fitted attributes EXCEPT 'lambda_'
            # which is a parameter name that scikit-learn forces us to use
            if key.endswith('_') and key != 'lambda_':
                continue
            params[key] = getattr(self, key)
        
        # Handle deep copying for nested estimators
        if deep:
            from sklearn.base import clone as sklearn_clone
            for key, value in list(params.items()):
                # Skip cloning certain parameters that sklearn will handle itself
                # or that contain raw H2O estimators (not our wrappers)
                if key in ('estimator_class', 'base_models'):
                    # estimator_class: it's a class reference, not an instance
                    # base_models: sklearn will handle cloning this list separately
                    continue
                # Only clone objects that have get_params and are not classes
                if hasattr(value, 'get_params') and not isinstance(value, type):
                    params[key] = sklearn_clone(value)
        
        return params

    def _get_param_names(cls):
        """Get parameter names for the estimator.
        
        This override is necessary because we use **kwargs in __init__.
        sklearn's default implementation only works with explicit parameters.
        """
        # For estimators using **kwargs, we need to return the actual parameters
        # that have been set on the instance
        if isinstance(cls, type):
            # Called as a class method - return empty list (shouldn't happen in practice)
            return []
        else:
            # Called on an instance - return the keys from get_params()
            return list(cls.get_params(deep=False).keys())

    def set_params(self, **params):
        """Sets the parameters of this estimator.

        This is a scikit-learn compatible set_params method that properly
        handles **kwargs-based initialization.
        """
        # First, handle the special 'lambda' -> 'lambda_' conversion
        if 'lambda' in params:
            params['lambda_'] = params.pop('lambda')

        # For each parameter, set it as an attribute
        # This is the correct approach for estimators using **kwargs
        for key, value in params.items():
            # Handle nested parameters (e.g., 'param__subparam')
            if '__' not in key:
                # Simple parameter - just set it
                setattr(self, key, value)
            else:
                # Nested parameter - use parent's set_params for proper handling
                super().set_params(**{key: value})
        
        return self