import inspect
import logging
import os
import tempfile
import uuid
import time
from typing import Any, Dict, List, Optional, Tuple

import h2o
import numpy as np
import pandas as pd
import psutil
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn
from sklearn.utils.validation import check_is_fitted

from ml_grid.util.global_params import global_parameters

# Configure logging
logger = logging.getLogger(__name__)

# Module-level shared checkpoint directory for all H2O classifier instances
# This ensures clones can access models trained by other instances
_SHARED_CHECKPOINT_DIR = tempfile.mkdtemp(prefix="h2o_checkpoints_")


class H2OBaseClassifier(BaseEstimator, ClassifierMixin):
    """Base class for scikit-learn compatible H2O classifier wrappers.

    Attributes:
        estimator_class: The H2O estimator class to be wrapped.
        logger: A logger instance for logging messages.
        model_ (Optional[Any]): The fitted H2O model object.
        classes_ (Optional[np.ndarray]): The class labels seen during fit.
        feature_names_ (Optional[list]): The names of the features seen during fit.
        feature_types_ (Optional[Dict[str, str]]): The H2O types of features.
        model_id (Optional[str]): The ID of the fitted H2O model.
    """

    MIN_SAMPLES_FOR_STABLE_FIT = 10

    # Class-level caches
    _init_param_names_cache = {}
    _estimator_signature_cache = {}

    _excluded_param_keys = {
        "estimator_class",
        "logger",
        "model",
        "model_id",
    }

    def __init__(self, estimator_class=None, **kwargs):
        """Initializes the H2OBaseClassifier.

        Args:
            estimator_class (Optional[type]): The H2O estimator class to be
                wrapped (e.g., `H2OGradientBoostingEstimator`).
            **kwargs: Additional keyword arguments to be passed to the H2O
                estimator during initialization.
        """
        # Handle estimator_class
        self.estimator_class = kwargs.pop("estimator_class", estimator_class)

        if not inspect.isclass(self.estimator_class):
            raise ValueError(
                "estimator_class is a required parameter and must be a class. "
                f"Received: {self.estimator_class}"
            )

        # Ensure lambda is stored as 'lambda_'
        if "lambda" in kwargs:
            kwargs["lambda_"] = kwargs.pop("lambda")

        # Set all kwargs as attributes for proper sklearn compatibility
        for key, value in kwargs.items():
            if key == "model":
                self.logger.warning(
                    "Rejecting 'model' parameter in __init__ - this conflicts with fitted attribute 'model_'"
                )
                continue
            setattr(self, key, value)

        # Initialize logger for this instance
        self.logger = logging.getLogger("ml_grid")

        # Internal state attributes
        self.model_: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list] = None
        self.feature_types_: Optional[Dict[str, str]] = None

        self._is_cluster_owner = False
        self._was_fit_on_constant_feature = False
        self._using_dummy_model = False
        self._rename_cols_on_predict = True

        # Use shared checkpoint directory across all clones
        self._checkpoint_dir = _SHARED_CHECKPOINT_DIR

        # H2O models are not safe with joblib's process-based parallelism.
        self._n_jobs = 1

        self._cached_param_names = None  # Cache for get_params

    # Class-level flag to track if H2O is initialized to avoid repeated API calls
    _h2o_initialized = False

    def __del__(self):
        """Cleans up the shared checkpoint directory if this is the last instance."""
        if hasattr(self, "_checkpoint_dir"):
            try:
                os.rmdir(self._checkpoint_dir)
            except OSError:
                pass

    def __getstate__(self):
        """Custom pickling to handle H2O models properly."""
        state = self.__dict__.copy()
        if "model_" in state:
            state["model_"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore state."""
        self.__dict__.update(state)

    def _ensure_h2o_is_running(self):
        """Safely checks for and initializes an H2O cluster if not running."""
        # Optimization: Check class-level flag first to avoid API overhead
        if H2OBaseClassifier._h2o_initialized:
            try:
                if h2o.connection().connected:
                    return
            except Exception:
                pass  # Fall through to full check

        try:
            cluster = h2o.cluster()
        except Exception:
            cluster = None

        show_progress = getattr(global_parameters, "h2o_show_progress", False)

        is_healthy = False
        if cluster and cluster.is_running():
            is_healthy = True
            try:
                # Check if cluster has memory
                memory = None
                try:
                    memory = cluster.total_mem()
                except Exception:
                    try:
                        memory = cluster.free_mem()
                    except Exception:
                        pass

                if memory is None:
                    self.logger.warning(
                        "H2O cluster memory check failed (None). Treating as unhealthy."
                    )
                    is_healthy = False
                elif isinstance(memory, (int, float)):
                    if memory < 1024 * 1024:  # < 1MB
                        self.logger.warning(
                            f"H2O cluster is running but reports {memory} memory. Treating as unhealthy."
                        )
                        is_healthy = False
            except Exception as e:
                self.logger.warning(f"H2O cluster check failed: {e}")
                is_healthy = False

        if not is_healthy:
            # If it was running but unhealthy, try to shut it down first to clear state
            if cluster and cluster.is_running():
                try:
                    self.logger.warning("Shutting down unhealthy H2O cluster...")
                    cluster.shutdown()
                    time.sleep(2)
                except Exception:
                    pass

            self.logger.info("Initializing H2O cluster...")

            # Get available system memory dynamically
            try:
                available_memory_bytes = psutil.virtual_memory().available
                memory_to_allocate_gb = int((available_memory_bytes * 0.8) / (1024**3))
                memory_to_allocate_gb = max(
                    1, memory_to_allocate_gb
                )  # Ensure at least 1GB

                self.logger.info(
                    f"Available system memory: {available_memory_bytes / (1024**3):.2f} GB"
                )
                self.logger.info(
                    f"Allocating {memory_to_allocate_gb} GB to H2O cluster (80% of available)"
                )

                h2o.init(
                    max_mem_size=f"{memory_to_allocate_gb}G",
                    nthreads=-1,
                    strict_version_check=False,
                )

                self.logger.info(
                    f"H2O cluster initialized successfully with {h2o.cluster().free_mem()} free memory"
                )

            except Exception as e:
                self.logger.warning(
                    f"Failed to allocate dynamic memory: {e}. Falling back to default initialization."
                )
                h2o.init(strict_version_check=False)

            self._is_cluster_owner = True

        H2OBaseClassifier._h2o_initialized = True

        # Set progress bar visibility based on the global parameter
        h2o.no_progress() if not show_progress else h2o.show_progress()

    def _validate_input_data(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validates and converts input data to proper format.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (Optional[pd.Series]): The target vector (optional, for fit-time validation).

        Returns:
            A tuple containing the validated DataFrame and Series.

        Raises:
            ValueError: If data is invalid
        """
        # Ensure X is a DataFrame and columns are strings (H2O requirement).
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_)
                if X.shape[1] != len(self.feature_names_):
                    raise ValueError(
                        f"Input data (X) has {X.shape[1]} columns, but expected {len(self.feature_names_)} "
                        f"based on training features. Please ensure column count matches."
                    )
            else:
                X = pd.DataFrame(X)
                X.columns = [str(c) for c in X.columns]
        else:
            if any(not isinstance(c, str) for c in X.columns):
                X.columns = X.columns.astype(str)

        # --- OPTIMIZATION: Skip validation if configured (saves ~99s in profile) ---
        if sklearn.get_config().get("skip_parameter_validation"):
            return X, y

        # Reset index to avoid sklearn CV indexing issues and maintain alignment
        if not isinstance(X.index, pd.RangeIndex):
            X = X.reset_index(drop=True)
            if y is not None and hasattr(y, "reset_index"):
                y = y.reset_index(drop=True)

        # Check for empty data
        if X.empty:
            raise ValueError("Cannot process empty DataFrame")

        # Validate y if provided (fit time)
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

            # Check for NaNs in the target variable
            if isinstance(y, pd.Series):
                has_nans = y.isnull().any()
            elif isinstance(y, pd.Categorical):
                has_nans = pd.isna(y.codes).any() or (y.codes == -1).any()
            else:
                try:
                    has_nans = np.isnan(y).any()
                except (TypeError, ValueError):
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
                unique_classes = pd.unique(y)

            if len(unique_classes) < 2:
                raise ValueError(
                    f"y must have at least 2 classes, found {len(unique_classes)}"
                )

        # Validate feature names match (predict time)
        if self.feature_names_ is not None and y is None:
            if list(X.columns) != self.feature_names_:
                missing_features = set(self.feature_names_) - set(X.columns)
                if missing_features:
                    raise ValueError(
                        f"Missing required features: {missing_features}. "
                        f"Expected: {self.feature_names_}, got: {list(X.columns)}"
                    )
                X = X[self.feature_names_]
                logger.debug("Reordered features to match training order")

        # Check for NaN values
        if X.isnull().any().any():
            raise ValueError(
                "Input data contains NaN values. Please handle missing values before fitting/predicting."
            )

        return X, y

    def _prepare_fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[h2o.H2OFrame, List[str], str, Dict[str, Any]]:
        """Prepares data and parameters for fitting.

        Returns:
            A tuple containing:
                - train_h2o (h2o.H2OFrame): The training data as an H2OFrame.
                - x_vars (List[str]): The list of feature column names.
                - outcome_var (str): The name of the outcome variable.
                - model_params (Dict[str, Any]): The dictionary of parameters for the H2O estimator.
        """
        if X.empty:
            raise ValueError(
                "Input data (X) is empty. This can happen during cross-validation with very small datasets. "
                "H2O models cannot be fitted on empty data."
            )
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")

        # Reset indices to ensure X and y are perfectly aligned
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.classes_ = np.unique(y)
        outcome_var = y.name
        x_vars = list(X.columns)

        # Store feature names for validation during predict
        self.feature_names_ = x_vars

        self.logger.debug(
            f">>> _prepare_fit: Set classes_={self.classes_}, feature_names_={self.feature_names_}"
        )

        self._ensure_h2o_is_running()

        # Convert target to pandas categorical type BEFORE creating H2OFrame
        y_categorical = pd.Categorical(y)
        y_series = pd.Series(y_categorical, name=outcome_var, index=y.index)

        if y_series.isnull().any():
            raise ValueError(
                "Target variable 'y' contains NaN values after processing. "
                "This is not supported by H2O models."
            )

        train_df = pd.concat([X, y_series], axis=1)

        try:
            train_h2o = h2o.H2OFrame(
                train_df, destination_frame=f"train_{uuid.uuid4().hex}"
            )
        except Exception as e:
            # Catch "Zero memory" error or other H2O server errors
            if "total cluster memory of Zero" in str(
                e
            ) or "IllegalArgumentException" in str(e):
                self.logger.warning(
                    f"H2OFrame creation failed: {e}. Attempting to restart H2O cluster."
                )

                # Force shutdown
                try:
                    h2o.cluster().shutdown()
                except Exception:
                    pass

                # Reset flag and wait
                H2OBaseClassifier._h2o_initialized = False
                time.sleep(3)

                # Re-initialize
                self._ensure_h2o_is_running()

                # Retry creation
                train_h2o = h2o.H2OFrame(
                    train_df, destination_frame=f"train_{uuid.uuid4().hex}"
                )
            else:
                raise e

        # Explicitly convert the outcome column to factor
        train_h2o[outcome_var] = train_h2o[outcome_var].asfactor()

        # Store column types to enforce them at predict time
        all_types = train_h2o.types
        self.feature_types_ = {col: all_types[col] for col in x_vars}

        # Get model parameters from instance attributes
        model_params = self._get_model_params()

        # Get valid parameters for the specific H2O estimator
        # Optimization: Use cached signature
        if self.estimator_class not in self._estimator_signature_cache:
            self._estimator_signature_cache[self.estimator_class] = inspect.signature(
                self.estimator_class
            ).parameters
        estimator_params = self._estimator_signature_cache[self.estimator_class]

        # If there's only one feature, prevent H2O from dropping it if it's constant
        if len(x_vars) == 1 and self.estimator_class:
            if X[x_vars[0]].nunique() <= 1:
                self._was_fit_on_constant_feature = True
                logger.warning(
                    "Fitting on a single constant feature - predictions may be unreliable"
                )

            if "ignore_const_cols" in estimator_params:
                model_params.setdefault("ignore_const_cols", False)

        # Conditionally add checkpoint directory
        if "export_checkpoints_dir" in estimator_params:
            model_params["export_checkpoints_dir"] = self._checkpoint_dir

        return train_h2o, x_vars, outcome_var, model_params

    def _get_model_params(self) -> Dict[str, Any]:
        """Extracts model parameters from instance attributes.

        Returns:
            A dictionary of parameters to pass to the H2O estimator.
        """
        all_params = {
            k: v
            for k, v in self.get_params(deep=False).items()
            if k != "estimator_class"
        }

        # Optimization: Use cached signature
        if self.estimator_class not in self._estimator_signature_cache:
            self._estimator_signature_cache[self.estimator_class] = inspect.signature(
                self.estimator_class
            ).parameters
        valid_param_keys = set(
            self._estimator_signature_cache[self.estimator_class].keys()
        )

        model_params = {
            key: value for key, value in all_params.items() if key in valid_param_keys
        }

        # Convert single-element numpy arrays/lists to native types for H2O
        for key, value in model_params.items():
            if isinstance(value, np.ndarray) and value.size == 1:
                model_params[key] = value.item()
            elif isinstance(value, list) and len(value) == 1:
                model_params[key] = value[0]

        return model_params

    def _handle_small_data_fallback(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Handles datasets smaller than MIN_SAMPLES_FOR_STABLE_FIT by fitting a dummy model.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.

        Returns:
            bool: True if a dummy model was fit, False otherwise.
        """
        if len(X) < self.MIN_SAMPLES_FOR_STABLE_FIT:
            self._using_dummy_model = True
            self.logger.warning(
                f"Small dataset ({len(X)} < {self.MIN_SAMPLES_FOR_STABLE_FIT}) - fitting dummy model."
            )
            self.classes_ = np.unique(y)  # Ensure classes_ is set even for dummy model
            return True
        return False

    def _sanitize_model_params(self):
        """Removes problematic parameters from the H2O model instance before training.

        Handles version mismatches (e.g., removing 'HGLM').
        """
        if self.model_ and hasattr(self.model_, "_parms"):
            if "HGLM" in self.model_._parms:
                self.logger.debug(
                    "Removing 'HGLM' parameter from H2O model to prevent backend error."
                )
                del self.model_._parms["HGLM"]

    def _has_constant_columns(self, X: pd.DataFrame) -> bool:
        """Checks for constant columns efficiently."""
        if X.empty:
            return False

        # Fast path for numeric data
        numeric_cols = X.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            mins = numeric_cols.min()
            maxs = numeric_cols.max()

            # Check 1: All NaNs (min is NaN)
            if mins.isna().any():
                return True

            # Check 2: Single value (min == max) AND No NaNs
            candidates = mins == maxs
            if candidates.any():
                candidate_cols = candidates[candidates].index
                # Check if any candidate has no NaNs
                if (numeric_cols[candidate_cols].count() == len(X)).any():
                    return True

        # Check remaining columns (objects, categoricals) using nunique
        other_cols = X.select_dtypes(exclude=np.number)
        if not other_cols.empty:
            if (other_cols.nunique(dropna=False) <= 1).any():
                return True

        return False

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "H2OBaseClassifier":
        """Fits the H2O model.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            H2OBaseClassifier: The fitted classifier instance.
        """
        try:
            self.logger.debug(f"=== fit() ENTRY on instance {id(self)} ===")
            self.logger.debug(
                f"Current attributes: {[k for k in self.__dict__.keys() if not k.startswith('_')]}"
            )

            if not hasattr(self, "estimator_class") or not self.estimator_class:
                raise AttributeError(
                    "H2OBaseClassifier is missing the 'estimator_class' attribute. "
                    "This typically happens if the scikit-learn cloning process is incomplete."
                )

            # Validate input data. This now returns a potentially modified X and y.
            X, y = self._validate_input_data(X, y)

            self.logger.debug("Calling _prepare_fit...")
            # Fit the actual model
            train_h2o, x_vars, outcome_var, model_params = self._prepare_fit(X, y)

            self.logger.debug(
                f"After _prepare_fit, classes_={getattr(self, 'classes_', 'MISSING')}, feature_names_={getattr(self, 'feature_names_', 'MISSING')}"
            )

            # Instantiate the H2O model with all the hyperparameters
            self.logger.debug(f"Creating H2O model with params: {model_params}")
            self.model_ = self.estimator_class(**model_params)

            # Sanitize parameters to prevent backend errors (e.g. HGLM)
            self._sanitize_model_params()

            # Force disable progress bar immediately before train
            if not getattr(global_parameters, "h2o_show_progress", False):
                h2o.no_progress()

            # Call the train() method with ONLY the data-related arguments
            self.logger.debug("Calling H2O model.train()...")
            self.model_.train(x=x_vars, y=outcome_var, training_frame=train_h2o)

            try:
                h2o.remove(train_h2o)
            except Exception:
                pass

            # Store model_id for recovery - THIS IS CRITICAL for predict() to work
            self.logger.debug(
                f"H2O train complete, extracting model_id from {self.model_}"
            )
            self.model_id = self.model_.model_id

            # Log for debugging
            self.logger.debug(
                f"Successfully fitted {self.estimator_class.__name__} with model_id: {self.model_id}"
            )
            self.logger.debug(
                f"Instance id: {id(self)}, has model_id: {hasattr(self, 'model_id')}, value: {getattr(self, 'model_id', 'MISSING')}"
            )
            self.logger.debug(
                f"Final attributes: {[k for k in self.__dict__.keys() if not k.startswith('_')]}"
            )

            return self

        except Exception as e:
            self.logger.error(
                f"EXCEPTION in fit() on instance {id(self)}: {e}", exc_info=True
            )
            self.logger.error(
                f"Attributes at exception: {[k for k in self.__dict__.keys() if not k.startswith('_')]}"
            )
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for samples in X.

        Args:
            X (pd.DataFrame): The feature matrix for prediction.

        Returns:
            np.ndarray: An array of predicted class labels.

        Raises:
            RuntimeError: If the model is not fitted or if prediction fails.
        """
        # Check that model was fitted
        try:
            check_is_fitted(self, ["model_id", "classes_", "feature_names_"])
        except Exception as e:
            # Add detailed debugging information
            self.logger.error(f"predict() called on unfitted instance {id(self)}")
            self.logger.error(f"  has model_id attr: {hasattr(self, 'model_id')}")
            self.logger.error(
                f"  model_id value: {getattr(self, 'model_id', 'MISSING')}"
            )
            self.logger.error(f"  has classes_ attr: {hasattr(self, 'classes_')}")
            self.logger.error(
                f"  has feature_names_ attr: {hasattr(self, 'feature_names_')}"
            )
            self.logger.error(
                f"  All attributes: {[k for k in self.__dict__.keys() if not k.startswith('_')]}"
            )
            raise RuntimeError(
                f"This H2OBaseClassifier instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator. "
                f"Details: {e}"
            )

        if self._was_fit_on_constant_feature:
            raise RuntimeError(
                "Predicting on a model that was fit on a single constant feature is unreliable. Halting."
            )

        # Validate input
        X, _ = self._validate_input_data(X)

        # Check for constant columns to prevent H2O backend crash
        if X.shape[1] > 0 and self._has_constant_columns(X):
            constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
            self.logger.warning(
                f"Prediction data contains constant columns: {constant_cols}. "
                "This can crash the H2O backend. Returning dummy predictions to prevent failure."
            )
            # Predict the first class as a fallback. This will result in a poor score for this fold,
            # which is the correct outcome for a degenerate test set.
            dummy_prediction = (
                self.classes_[0]
                if self.classes_ is not None and len(self.classes_) > 0
                else 0
            )
            return np.full(len(X), dummy_prediction)

        # Lazy load model and optimistically assume H2O is running
        if self.model_ is None:
            self._ensure_h2o_is_running()
            self._ensure_model_is_loaded()

        try:
            # Pass column_types directly to constructor to avoid expensive casting loop
            col_types = None
            if self.feature_types_:
                col_types = {
                    k: v for k, v in self.feature_types_.items() if k in X.columns
                }

            tmp_frame = h2o.H2OFrame(
                X,
                column_names=self.feature_names_,
                column_types=col_types,
                destination_frame=f"pred_{uuid.uuid4().hex}",
            )

            test_h2o = tmp_frame

        except Exception as e:
            if isinstance(e, TimeoutError):
                raise
            raise RuntimeError(f"Failed to create H2O frame for prediction: {e}")

        # Make prediction
        try:
            predictions = self.model_.predict(test_h2o)
        except Exception as e:
            # If prediction failed, it might be because the model was unloaded/GC'd on server.
            if isinstance(e, TimeoutError):
                raise
            # Try reloading and predicting again.
            self.logger.debug(f"Prediction failed ({e}), attempting to reload model...")
            try:
                self._ensure_h2o_is_running()
                self._ensure_model_is_loaded()
                predictions = self.model_.predict(test_h2o)
            except Exception as e2:
                if isinstance(e2, TimeoutError):
                    raise
                # Catch H2O backend crashes (NPE) during prediction and fallback
                if "java.lang.NullPointerException" in str(e):
                    self.logger.warning(
                        f"H2O backend crashed with NPE during predict(). Returning dummy predictions. Details: {e}"
                    )
                    # Fallback: predict the first class (usually 0)
                    dummy_val = (
                        self.classes_[0]
                        if self.classes_ is not None and len(self.classes_) > 0
                        else 0
                    )
                    return np.full(len(X), dummy_val)

                raise RuntimeError(f"H2O prediction failed: {e2}")

        # Extract predictions
        pred_df = predictions.as_data_frame(use_multi_thread=False)

        try:
            h2o.remove(test_h2o)
            h2o.remove(predictions)
        except Exception:
            pass

        if "predict" in pred_df.columns:
            preds = pred_df["predict"].values.ravel()
            # Cast to the same type as classes_ to avoid object/string overhead
            if self.classes_ is not None:
                try:
                    # Ensure we are casting to a native numpy type, not object
                    dtype = self.classes_.dtype
                    if dtype == np.object_:
                        preds = preds.astype(str)
                    else:
                        preds = preds.astype(dtype)
                except (ValueError, TypeError):
                    pass
            return preds
        else:
            raise RuntimeError("Prediction output missing 'predict' column")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for samples in X.

        Args:
            X (pd.DataFrame): The feature matrix for prediction.

        Returns:
            np.ndarray: An array of shape (n_samples, n_classes) with class probabilities.

        Raises:
            RuntimeError: If the model is not fitted or if prediction fails.
        """
        try:
            check_is_fitted(
                self, ["model_id", "classes_", "feature_names_", "feature_types_"]
            )
        except Exception as e:
            raise RuntimeError(
                f"This H2OBaseClassifier instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator. "
                f"Details: {e}"
            )

        if self._was_fit_on_constant_feature:
            raise RuntimeError(
                "Predicting probabilities on a model that was fit on a single constant feature is unreliable. Halting."
            )

        # Validate input
        X, _ = self._validate_input_data(X)

        if X.shape[1] > 0 and self._has_constant_columns(X):
            constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
            self.logger.warning(
                f"Prediction data contains constant columns: {constant_cols}. "
                "This can crash the H2O backend. Returning dummy probabilities to prevent failure."
            )
            # Return a uniform probability distribution.
            n_classes = (
                len(self.classes_)
                if self.classes_ is not None and len(self.classes_) > 0
                else 2
            )
            dummy_probas = np.full((len(X), n_classes), 1 / n_classes)
            return dummy_probas

        # Lazy load model and optimistically assume H2O is running
        if self.model_ is None:
            self._ensure_h2o_is_running()
            self._ensure_model_is_loaded()

        # Create H2O frame with explicit column names
        try:
            col_types = None
            if self.feature_types_:
                col_types = {
                    k: v for k, v in self.feature_types_.items() if k in X.columns
                }

            test_h2o = h2o.H2OFrame(
                X,
                column_names=self.feature_names_,
                column_types=col_types,
                destination_frame=f"prob_{uuid.uuid4().hex}",
            )
        except Exception as e:
            if isinstance(e, TimeoutError):
                raise
            raise RuntimeError(f"Failed to create H2O frame for prediction: {e}")

        # Make prediction
        try:
            predictions = self.model_.predict(test_h2o)
        except Exception as e:
            # Retry logic for unloaded models
            if isinstance(e, TimeoutError):
                raise
            self.logger.debug(f"Prediction failed ({e}), attempting to reload model...")
            try:
                self._ensure_h2o_is_running()
                self._ensure_model_is_loaded()
                predictions = self.model_.predict(test_h2o)
            except Exception as e2:
                if isinstance(e2, TimeoutError):
                    raise
                if "java.lang.NullPointerException" in str(e):
                    self.logger.warning(
                        f"H2O backend crashed with NPE during predict_proba(). Returning dummy probabilities. Details: {e}"
                    )
                    # Fallback: uniform probabilities
                    n_classes = (
                        len(self.classes_)
                        if self.classes_ is not None and len(self.classes_) > 0
                        else 2
                    )
                    return np.full((len(X), n_classes), 1.0 / n_classes)

                raise RuntimeError(f"H2O prediction failed: {e2}")

        # Extract probabilities (drop the 'predict' column)
        res_df = predictions.as_data_frame(use_multi_thread=False)
        if "predict" in res_df.columns:
            prob_df = res_df.drop(columns=["predict"])
        else:
            prob_df = res_df

        try:
            h2o.remove(test_h2o)
            h2o.remove(predictions)
        except Exception:
            pass

        # Ensure strictly float64 to avoid object overhead in sklearn metrics
        return prob_df.values.astype(np.float64)

    def _ensure_model_is_loaded(self):
        """
        Ensures the H2O model is loaded into memory, reloading from checkpoint if necessary.
        This prevents errors when the H2O cluster garbage collects the model.
        """
        # This should never happen if predict() does its job, but defensive check
        if self.model_id is None:
            raise RuntimeError(
                "Cannot load model: model_id is not set. Model may not have been fitted. "
                "This is an internal error - please ensure fit() was called successfully."
            )

        # If model_ is already loaded, check if it's still valid in H2O
        if self.model_ is not None:
            try:
                # Quick test: try to get model from cluster by ID
                h2o.get_model(self.model_id)
                # If we got here, model is still in cluster and model_ is valid
                return
            except Exception:
                # Model was garbage collected, need to reload
                self.logger.debug(
                    f"Model {self.model_id} was garbage collected, reloading..."
                )
                self.model_ = None

        # Try to get the model from H2O cluster
        try:
            self.model_ = h2o.get_model(self.model_id)
            self.logger.debug(
                f"Successfully retrieved model {self.model_id} from H2O cluster"
            )
            return
        except Exception as e:
            self.logger.warning(f"Model {self.model_id} not found in H2O cluster: {e}")

        # If not in cluster, try to reload from checkpoint
        checkpoint_path = os.path.join(self._checkpoint_dir, self.model_id)
        self.logger.debug(
            f"Attempting to reload model from checkpoint: {checkpoint_path}"
        )

        if os.path.exists(checkpoint_path):
            try:
                self.model_ = h2o.load_model(checkpoint_path)
                self.logger.info(
                    f"Successfully reloaded model {self.model_id} from checkpoint."
                )
                return
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reload model {self.model_id} from checkpoint {checkpoint_path}: {e}"
                )
        else:
            # List what's actually in the checkpoint directory for debugging
            try:
                available_files = (
                    os.listdir(self._checkpoint_dir)
                    if os.path.exists(self._checkpoint_dir)
                    else []
                )
            except Exception:
                available_files = ["<error listing directory>"]

            raise RuntimeError(
                f"Fatal: Model {self.model_id} not found in H2O cluster and no checkpoint exists at {checkpoint_path}. "
                f"Checkpoint directory: {self._checkpoint_dir}. "
                f"Available files: {available_files}"
            )

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle H2O models properly."""
        # Create a new instance with the same parameters
        params = self.get_params(deep=False)
        cloned = self.__class__(**params)

        # Copy over fitted attributes if they exist
        for attr in [
            "model_id",
            "classes_",
            "feature_names_",
            "_was_fit_on_constant_feature",
        ]:
            if hasattr(self, attr):
                setattr(cloned, attr, getattr(self, attr))
                self.logger.debug(
                    f"__deepcopy__: copied {attr} = {getattr(self, attr)}"
                )

        # Don't copy model_ - it will be reloaded from checkpoint
        cloned.model_ = None

        self.logger.debug(
            f"__deepcopy__ called: original {id(self)}, clone {id(cloned)}"
        )
        return cloned

    def _get_param_names(self):
        """Get parameter names for the estimator.

        This override is necessary because we use **kwargs in __init__.
        It's an instance method to access parameters stored on self.

        This should ONLY return parameter names, NOT fitted attribute names.
        """
        if self._cached_param_names is not None:
            return self._cached_param_names

        # Optimization: Cache the signature inspection on the class
        cls = self.__class__
        if cls not in self._init_param_names_cache:
            init_signature = inspect.signature(cls.__init__)
            self._init_param_names_cache[cls] = [
                p.name
                for p in init_signature.parameters.values()
                if p.name not in ("self", "args", "kwargs")
            ]

        init_params = self._init_param_names_cache[cls]

        # Optimization: Use sets for O(1) lookup
        init_params_set = set(init_params)

        extra_params = [
            key
            for key in self.__dict__
            if not key.startswith("_")
            and not (key.endswith("_") and key != "lambda_")  # Allow lambda_
            and key not in init_params_set
            and key not in self._excluded_param_keys
        ]

        self._cached_param_names = sorted(init_params + extra_params)
        return self._cached_param_names

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Optimized implementation that bypasses BaseEstimator.get_params overhead.
        """
        # Optimization: Directly construct dict from cached param names
        # We bypass the deep=True recursion check for speed, as H2O estimators
        # don't have nested sklearn estimators.
        return {key: getattr(self, key) for key in self._get_param_names()}

    def set_params(self: "H2OBaseClassifier", **kwargs: Any) -> "H2OBaseClassifier":
        """Sets the parameters of this estimator, compatible with scikit-learn.

        Args:
            **kwargs: Keyword arguments representing the parameters to set.
        Returns:
            H2OBaseClassifier: The classifier instance with updated parameters.
        """
        self.logger.debug(
            f">>> set_params() called on instance {id(self)} with keys: {list(kwargs.keys())}"
        )

        # Handle lambda -> lambda_ conversion
        if "lambda" in kwargs:
            kwargs["lambda_"] = kwargs.pop("lambda")

        # Preserve fitted attributes
        fitted_attributes = {}
        for attr in ["model_", "classes_", "feature_names_", "model_id"]:
            if hasattr(self, attr):
                fitted_attributes[attr] = getattr(self, attr)
                self.logger.debug(f"Preserving fitted attribute: {attr}")

        # Reject any attempts to set 'model' or other fitted-like attributes
        forbidden_keys = ["model", "model_", "classes_", "feature_names_", "model_id"]
        for key in list(kwargs.keys()):
            if key in forbidden_keys:
                self.logger.warning(f"Rejecting forbidden key in set_params: '{key}'")
                kwargs.pop(key)

        # Set each parameter as an attribute
        for key, value in kwargs.items():
            if "__" not in key:
                setattr(self, key, value)
            else:
                # This shouldn't happen for our use case, but handle it anyway
                setattr(self, key, value)

        self._cached_param_names = None  # Invalidate cache

        # Restore fitted attributes
        for attr, value in fitted_attributes.items():
            setattr(self, attr, value)
            self.logger.debug(f"Restored fitted attribute: {attr}")

        self.logger.debug(
            f">>> set_params() complete. Final keys: {list(self.__dict__.keys())}"
        )
        return self
