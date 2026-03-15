import time
import logging
import multiprocessing
import joblib
import warnings
import functools
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import sklearn

# --- FIX: Lazy import torch only when needed ---
try:
    import torch
except ImportError:
    torch = None

from sklearn import metrics

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    ParameterGrid,
    RepeatedKFold,
    KFold,
    cross_validate,
)

from ml_grid.model_classes.keras_classifier_class import KerasClassifierClass
from ml_grid.pipeline.hyperparameter_search import HyperparameterSearch
from ml_grid.util.debug_print_statements import debug_print_statements_class
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class
from ml_grid.util.validate_parameters import validate_parameters_helper
from ml_grid.util.bayes_utils import is_skopt_space
from skopt.space import Categorical

# Global flag to ensure TensorFlow/GPU setup runs only once per process
try:
    from aeon.classification.deep_learning.base import BaseDeepClassifier
except (ImportError, ModuleNotFoundError):
    BaseDeepClassifier = None  # Allows code to run without aeon

_TF_INITIALIZED = False


# Disable TF Traceback Filtering to reduce overhead in Keras model building
try:
    tf.debugging.disable_traceback_filtering()
except (AttributeError, ImportError):
    pass


def _patch_aeon_models():
    """
    Patches aeon classifiers to fix state-related issues during cloning, prediction,
    and parameter conflicts (e.g. MUSE), which affect sklearn search tools.
    """
    try:
        from aeon.classification.base import BaseClassifier
        from aeon.classification.deep_learning.base import BaseDeepClassifier

        # The error `AttributeError: '...' object has no attribute '_metrics'` happens
        # because sklearn.clone() creates a fresh object, losing internal state.
        # The `fit` method is supposed to re-initialize `_metrics` from `metrics`,
        # but this seems to fail intermittently inside the complex search-cv process.
        #
        # This patch wraps the `fit` method to ensure `_metrics` is present
        # *before* the original `fit` logic is called. This acts as a safeguard.

        def _prepare_deep_learning_data(X, min_length=128):
            """
            Prepares data for aeon deep learning models by ensuring it is in the
            format Keras expects (N, T, C) and padding the timepoints dimension to
            a minimum length. A larger min_length is recommended to accommodate
            hyperparameter searches that may result in deep networks with large
            receptive fields (e.g., from high dilation rates in deeper layers),
            which can cause 'negative output size' errors even if padding seems
            sufficient for the first layer.
            """
            # Ensure X is a numpy array; aeon might pass lists or pandas objects
            if not isinstance(X, np.ndarray):
                try:
                    X = np.array(X)
                except Exception:
                    return X

            # Convert 2D (N, T) to 3D (N, C=1, T) for consistent handling
            if X.ndim == 2:
                X = np.expand_dims(X, axis=1)

            if X.ndim == 3:
                # Robustly pad both dimensions 1 and 2 if they are too small.
                # This handles ambiguity between (N, C, T) and (N, T, C) layouts
                # and prevents "Computed output size would be negative" crashes
                # caused by dimension size 0 or very small values.
                for axis in [1, 2]:
                    if X.shape[axis] < min_length:
                        pad_width = min_length - X.shape[axis]
                        logging.getLogger("ml_grid").debug(
                            f"Padding axis {axis} from {X.shape[axis]} to {min_length}"
                        )

                        pad_config = [(0, 0), (0, 0), (0, 0)]
                        pad_config[axis] = (0, pad_width)

                        # Use 'constant' padding (zeros) if dimension is empty (size 0),
                        # otherwise use 'edge' to repeat last value.
                        # 'edge' raises ValueError on size 0 arrays.
                        mode = "constant" if X.shape[axis] == 0 else "edge"

                        X = np.pad(X, tuple(pad_config), mode=mode)

                # Transpose from (N, C, T) to (N, T, C) for Keras channels_last default.
                # This ensures Keras interprets T as Steps and C as Channels.
                X = np.transpose(X, (0, 2, 1))

            return X

        if not getattr(BaseClassifier, "_mlgrid_patched_fit", False):
            original_fit = BaseClassifier.fit

            @functools.wraps(original_fit)
            def patched_fit(self, X, y):
                # This patch fixes an `AttributeError` for `_metrics` in aeon deep
                # learning models when used with sklearn's hyperparameter search.
                # The search process clones the model, and the internal `_metrics`
                # attribute is not correctly synchronized with the public `metrics`
                # parameter, causing a crash in `build_model`.
                # This patch unconditionally sets `_metrics` from `metrics` at the
                # start of every `fit` call to ensure the model's state is correct.
                metrics_val = getattr(self, "metrics", "accuracy")
                # Ensure `_metrics` is always a list, as Keras expects.
                self._metrics = (
                    [metrics_val]
                    if not isinstance(metrics_val, (list, tuple))
                    else list(metrics_val)
                )

                # Fix for ResNet shape mismatch when padding != 'same'
                # aeon's ResNet implementation has a bug where the shortcut layer (stride 1, 1x1 conv)
                # does not match the output shape of the residual block if padding is 'valid' (reducing size).
                # We force 'same' padding to ensure shapes align for the Add() layer.
                if "ResNet" in self.__class__.__name__:
                    if hasattr(self, "padding") and self.padding != "same":
                        logging.getLogger("ml_grid").warning(
                            f"Forcing padding='same' on {self.__class__.__name__} to prevent ResNet shortcut mismatch."
                        )
                        self.padding = "same"

                    # Fix for ResNet parameter mismatch when n_conv_per_residual_block is tuned
                    # but kernel_size/strides/dilation_rate are default lists (e.g. len 3).
                    if hasattr(self, "n_conv_per_residual_block"):
                        n_conv = self.n_conv_per_residual_block

                        # Handle default kernel_size (None) which defaults to [8, 5, 3] in aeon.
                        # If n_conv is changed, we must materialize the default and resize it.
                        if (
                            hasattr(self, "kernel_size")
                            and getattr(self, "kernel_size") is None
                        ):
                            self.kernel_size = [8, 5, 3]

                        for param in ["kernel_size", "strides", "dilation_rate"]:
                            if hasattr(self, param):
                                val = getattr(self, param)
                                if (
                                    isinstance(val, (list, tuple))
                                    and len(val) != n_conv
                                ):
                                    is_tuple = isinstance(val, tuple)
                                    val_list = list(val)
                                    logging.getLogger("ml_grid").warning(
                                        f"Adjusting ResNet {param} length from {len(val_list)} to {n_conv} to match n_conv_per_residual_block."
                                    )
                                    if len(val_list) > n_conv:
                                        new_val = val_list[:n_conv]
                                    else:
                                        new_val = val_list + [val_list[-1]] * (
                                            n_conv - len(val_list)
                                        )

                                    setattr(
                                        self,
                                        param,
                                        tuple(new_val) if is_tuple else new_val,
                                    )

                # Check for Deep Learning models and pad input if too short
                # This prevents "ValueError: Computed output size would be negative" in Keras
                # when using models like ResNet or InceptionTime on very short sequences.
                if isinstance(self, BaseDeepClassifier):
                    X = _prepare_deep_learning_data(X)

                return original_fit(self, X, y)

            BaseClassifier.fit = patched_fit
            setattr(BaseClassifier, "_mlgrid_patched_fit", True)

    except (ImportError, AttributeError):
        pass  # aeon not installed

    # --- NEW PATCH for predict method ---
    # This ensures that data passed to `predict` is padded just like in `fit`,
    # resolving "X has different length" errors for deep learning models.
    try:
        from aeon.classification.base import BaseClassifier

        if not getattr(BaseClassifier, "_mlgrid_patched_predict", False):
            original_predict = BaseClassifier.predict

            @functools.wraps(original_predict)
            def patched_predict(self, X):
                # Only apply padding if the model seems to be a deep learning one
                # that was padded during fit. Use robust isinstance check.
                if isinstance(self, BaseDeepClassifier):
                    X = _prepare_deep_learning_data(X)
                return original_predict(self, X)

            BaseClassifier.predict = patched_predict
            setattr(BaseClassifier, "_mlgrid_patched_predict", True)
    except (ImportError, AttributeError):
        pass

    # --- NEW PATCH for public predict_proba method ---
    # This ensures that data passed to `predict_proba` is padded,
    # resolving "X has different length" errors for deep learning models.
    try:
        from aeon.classification.base import BaseClassifier

        if not getattr(BaseClassifier, "_mlgrid_patched_public_predict_proba", False):
            original_public_predict_proba = BaseClassifier.predict_proba

            @functools.wraps(original_public_predict_proba)
            def patched_public_predict_proba(self, X, **kwargs):
                # Only apply padding if the model is a deep learning one.
                if isinstance(self, BaseDeepClassifier):
                    X = _prepare_deep_learning_data(X)
                return original_public_predict_proba(self, X, **kwargs)

            BaseClassifier.predict_proba = patched_public_predict_proba
            setattr(BaseClassifier, "_mlgrid_patched_public_predict_proba", True)
    except (ImportError, AttributeError):
        pass

    # --- MODIFIED PATCH for internal _predict_proba ---
    # This patch now ONLY handles NaN values. Padding is handled by the public
    # method wrappers (fit, predict, predict_proba) to prevent double-padding.
    try:
        from aeon.classification.deep_learning.base import BaseDeepClassifier

        if not getattr(BaseDeepClassifier, "_mlgrid_patched_predict_proba", False):
            original_predict_proba = BaseDeepClassifier._predict_proba

            @functools.wraps(original_predict_proba)
            def patched_internal_predict_proba(self, X):
                # Call the original method to get probabilities
                # Padding is now handled by the public wrappers (predict/predict_proba)
                y_pred_proba = original_predict_proba(self, X)

                # Check for NaNs, which indicate unstable training
                if np.isnan(y_pred_proba).any():
                    logging.getLogger("ml_grid").warning(
                        "Model produced NaN probabilities, indicating instability. "
                        "Replacing with uniform distribution to avoid crash."
                    )
                    # Find rows with NaNs
                    nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
                    # Replace NaN rows with a uniform probability distribution
                    n_classes = len(self.classes_)
                    y_pred_proba[nan_rows] = 1.0 / n_classes

                return y_pred_proba

            BaseDeepClassifier._predict_proba = patched_internal_predict_proba
            setattr(BaseDeepClassifier, "_mlgrid_patched_predict_proba", True)
            logging.getLogger("ml_grid").info(
                "Applied NaN-proofing patch to aeon's BaseDeepClassifier._predict_proba."
            )

    except (ImportError, AttributeError) as e:
        logging.getLogger("ml_grid").debug(
            f"Could not apply aeon predict_proba patch: {e}"
        )
        pass

    # --- NEW PATCH for MUSE: handles param conflicts and IndexErrors ---
    # Fixes two issues:
    # 1. ValueError: "Please set either variance or anova" when both are True.
    # 2. IndexError: when no SFA words are generated (e.g., from flat data),
    #    preventing the model from fitting.
    try:
        from aeon.classification.dictionary_based import MUSE
        from sklearn.dummy import DummyClassifier

        if not getattr(MUSE, "_mlgrid_patched_muse_fit", False):
            original_muse_fit = MUSE._fit

            @functools.wraps(original_muse_fit)
            def patched_muse__fit(self, X, y):
                # 1. Fix variance/anova conflict
                if getattr(self, "variance", False) and getattr(self, "anova", False):
                    logging.getLogger("ml_grid").warning(
                        "MUSE: Both 'variance' and 'anova' are True. Setting 'anova' to False to prevent crash."
                    )
                    self.anova = False

                # 2. Fix min_window > max_window for short series
                n_timepoints = X.shape[-1]
                # Replicate aeon's internal default for max_window if it's None
                effective_max_window = (
                    self.max_window if self.max_window is not None else n_timepoints
                )

                if self.min_window > effective_max_window:
                    logging.getLogger("ml_grid").warning(
                        f"MUSE: min_window ({self.min_window}) > max_window ({effective_max_window}). "
                        f"Adjusting min_window to {effective_max_window} to prevent crash."
                    )
                    self.min_window = effective_max_window
                if self.min_window < 1:
                    self.min_window = 1

                try:
                    # 3. Attempt to run the original fit
                    return original_muse_fit(self, X, y)
                except IndexError:
                    # 4. If it fails with an IndexError (from empty `all_words`), fit a dummy model
                    logging.getLogger("ml_grid").warning(
                        "MUSE._fit failed with IndexError. This often means no features were extracted. Fitting a dummy classifier."
                    )
                    self.clf = DummyClassifier(strategy="most_frequent")
                    self.clf.fit(np.zeros((len(y), 1)), y)
                    return self

            MUSE._fit = patched_muse__fit
            setattr(MUSE, "_mlgrid_patched_muse_fit", True)
    except (ImportError, AttributeError):
        pass

    # --- NEW PATCH for MUSE._transform_words ---
    # This prevents crashes during predict/predict_proba when no features
    # can be extracted from the input data (e.g., from flat test data).
    try:
        from aeon.classification.dictionary_based import MUSE

        if not getattr(MUSE, "_mlgrid_patched_transform_words", False):
            original_transform_words = MUSE._transform_words

            @functools.wraps(original_transform_words)
            def patched_transform_words(self, X):
                try:
                    # Attempt to run the original transformation
                    return original_transform_words(self, X)
                except IndexError:
                    # This occurs when no SFA words are generated.
                    logging.getLogger("ml_grid").warning(
                        "MUSE._transform_words failed with IndexError. This often means no features could be extracted from the prediction data. Returning a zero-vector."
                    )
                    # Determine the number of features the internal classifier expects.
                    n_features = getattr(self.clf, "n_features_in_", 1)
                    n_instances = X.shape[0]
                    # Return a zero matrix of the correct shape.
                    return np.zeros((n_instances, n_features))

            MUSE._transform_words = patched_transform_words
            setattr(MUSE, "_mlgrid_patched_transform_words", True)
    except (ImportError, AttributeError):
        pass

    # --- NEW PATCH for OrdinalTDE NaN predictions ---
    # Fixes ValueError: "'a' cannot be empty unless no samples are taken" in _predict
    # when _predict_proba returns NaNs.
    try:
        try:
            from aeon.classification.ordinal_classification import OrdinalTDE
        except ImportError:
            from aeon.classification.ordinal_classification._ordinal_tde import (
                OrdinalTDE,
            )

        if not getattr(OrdinalTDE, "_mlgrid_patched_predict_proba", False):
            original_tde_predict_proba = OrdinalTDE._predict_proba

            @functools.wraps(original_tde_predict_proba)
            def patched_tde_predict_proba(self, X):
                y_pred_proba = original_tde_predict_proba(self, X)
                if np.isnan(y_pred_proba).any():
                    logging.getLogger("ml_grid").warning(
                        "OrdinalTDE produced NaN probabilities. Replacing with uniform distribution."
                    )
                    nan_rows = np.any(np.isnan(y_pred_proba), axis=1)
                    y_pred_proba[nan_rows] = 1.0 / len(self.classes_)
                return y_pred_proba

            OrdinalTDE._predict_proba = patched_tde_predict_proba
            setattr(OrdinalTDE, "_mlgrid_patched_predict_proba", True)
    except (ImportError, AttributeError):
        pass

    # --- NEW PATCH for IndividualInceptionClassifier __init__ ---
    # This is a surgical fix for the recurring `AttributeError: '_metrics' missing`
    # in the InceptionTime ensemble's sub-classifiers. Patching __init__ ensures
    # the attribute is present from the moment of instantiation, which is more
    # reliable than patching the `fit` method for these internal models.
    try:
        from aeon.classification.deep_learning._inception_time import (
            IndividualInceptionClassifier,
        )

        if not getattr(IndividualInceptionClassifier, "_mlgrid_patched_init", False):
            original_init = IndividualInceptionClassifier.__init__

            @functools.wraps(original_init)
            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                # After the original init, force-set _metrics to prevent AttributeError
                metrics_val = getattr(self, "metrics", "accuracy")
                if metrics_val is None:
                    self._metrics = []
                elif isinstance(metrics_val, str):
                    self._metrics = [metrics_val]
                else:
                    self._metrics = list(metrics_val)

                # Clone the optimizer to ensure a fresh instance for each individual classifier.
                # This prevents the Keras "Unknown variable" error when the same optimizer
                # instance is reused across different models in the ensemble.
                if hasattr(self, "optimizer") and isinstance(
                    self.optimizer, tf.keras.optimizers.Optimizer
                ):
                    self.optimizer = self.optimizer.from_config(
                        self.optimizer.get_config()
                    )

            IndividualInceptionClassifier.__init__ = patched_init
            setattr(IndividualInceptionClassifier, "_mlgrid_patched_init", True)
    except (ImportError, AttributeError):
        pass

    # --- NEW PATCH for SummaryClassifier ---
    # Fixes ValueError: "Summary function input (...) not recognised"
    # This happens when a tuple of stats is passed (e.g. via hyperparameter search)
    # but aeon only accepts specific string values like "default" or "percentiles".
    try:
        from aeon.classification.feature_based import SummaryClassifier

        if not getattr(SummaryClassifier, "_mlgrid_patched_summary_fit", False):
            original_summary_fit = SummaryClassifier._fit

            @functools.wraps(original_summary_fit)
            def patched_summary_fit(self, X, y):
                # Valid options for aeon's SevenNumberSummary
                valid_options = ["default", "percentiles"]

                # Check and fix self.summary_stats
                if hasattr(self, "summary_stats"):
                    if self.summary_stats not in valid_options:
                        logging.getLogger("ml_grid").warning(
                            f"SummaryClassifier: summary_stats '{self.summary_stats}' not recognised by aeon. "
                            "Resetting to 'default' to prevent crash."
                        )
                        self.summary_stats = "default"

                # Check and fix self.transformer_.summary_stats if it exists (it likely does)
                # We must update the transformer because it might have been initialized with the bad value
                if hasattr(self, "transformer_") and hasattr(
                    self.transformer_, "summary_stats"
                ):
                    if self.transformer_.summary_stats not in valid_options:
                        self.transformer_.summary_stats = "default"

                return original_summary_fit(self, X, y)

            SummaryClassifier._fit = patched_summary_fit
            setattr(SummaryClassifier, "_mlgrid_patched_summary_fit", True)
    except (ImportError, AttributeError):
        pass


class grid_search_crossvalidate_ts:
    def __init__(
        self,
        algorithm_implementation: Any,
        parameter_space: Union[Dict, List[Dict]],
        method_name: str,
        ml_grid_object: Any,
        sub_sample_parameter_val: int = 100,
        project_score_save_class_instance: Optional[project_score_save_class] = None,
    ):
        """Initializes and runs a cross-validated hyperparameter search for Time Series models.

        This class is optimized for time-series models, expecting NumPy arrays instead of
        pandas DataFrames and removing logic for standard tabular models (H2O, FLAML, etc.).

        Args:
            algorithm_implementation (Any): The scikit-learn compatible estimator
                instance.
            parameter_space (Union[Dict, List[Dict]]): The dictionary or list of
                dictionaries defining the hyperparameter search space.
            method_name (str): The name of the algorithm method.
            ml_grid_object (Any): The main pipeline object containing all data
                (X_train, y_train, etc.) and parameters for the current
                iteration.
            sub_sample_parameter_val (int, optional): A value used to limit
                the number of iterations in a randomized search. Defaults to 100.
            project_score_save_class_instance (Optional[project_score_save_class], optional):
                An instance of the score saving class. Defaults to None.
        """
        # Set warning filters
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.logger = logging.getLogger("ml_grid")

        self.global_params = global_parameters

        self.verbose = self.global_params.verbose

        self.project_score_save_class_instance = project_score_save_class_instance

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        random_grid_search = self.global_params.random_grid_search

        self.sub_sample_parameter_val = sub_sample_parameter_val

        # Patch aeon models to fix pickling/state issues
        _patch_aeon_models()

        logging.debug(f"Methods name: {method_name}")
        logging.debug(f"Parameter space: {parameter_space}")
        logging.debug(f"Algorithm implementation: {algorithm_implementation}")

        # Detect nested parallelism: force n_jobs=1 if running inside a worker process
        if multiprocessing.current_process().daemon:
            self.global_params.grid_n_jobs = 1
            grid_n_jobs = 1
        else:
            grid_n_jobs = self.global_params.grid_n_jobs

        # Configure GPU usage and job limits for specific models
        is_gpu_model = (
            "keras" in method_name.lower()
            or "neural" in method_name.lower()
            or "torch" in method_name.lower()
            or "inception" in method_name.lower()
            or "fcn" in method_name.lower()
            or "tapnet" in method_name.lower()
            or "encoder" in method_name.lower()
            or "resnet" in method_name.lower()
            or "cnn" in method_name.lower()
            or "mlp" in method_name.lower()
        )

        global _TF_INITIALIZED
        if is_gpu_model:
            grid_n_jobs = 1

            # One-time TF/GPU Setup
            if is_gpu_model and not _TF_INITIALIZED:
                try:
                    # --- FIX for libdevice error ---
                    if "XLA_FLAGS" not in os.environ:
                        site_packages_path = next(
                            (p for p in sys.path if "site-packages" in p), None
                        )
                        if site_packages_path:
                            cuda_path = os.path.join(
                                site_packages_path, "nvidia", "cuda_nvcc"
                            )

                            if os.path.exists(cuda_path):
                                self.logger.info(
                                    f"Found CUDA compiler toolkit at {cuda_path}. Setting XLA_FLAGS."
                                )
                                os.environ["XLA_FLAGS"] = (
                                    f"--xla_gpu_cuda_data_dir={cuda_path}"
                                )
                            else:
                                self.logger.warning(
                                    "Could not find 'nvidia/cuda_nvcc' directory. Falling back to site-packages root. "
                                    "Install 'nvidia-cuda-nvcc-cu12' for a reliable setup."
                                )

                    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
                    if gpu_devices:
                        for device in gpu_devices:
                            try:
                                tf.config.experimental.set_memory_growth(device, True)
                            except RuntimeError:
                                pass
                    else:
                        tf.config.set_visible_devices([], "GPU")

                    # --- FIX for NotImplementedError ---
                    # Force eager execution to resolve "numpy() is only available when
                    # eager execution is enabled" errors, which occur when a library
                    # (like aeon) tries to inspect a tensor's value during graph execution.
                    tf.config.run_functions_eagerly(True)
                except Exception as e:
                    self.logger.warning(f"Could not configure GPU for TensorFlow: {e}")
                finally:
                    _TF_INITIALIZED = True

        self.metric_list = self.global_params.metric_list

        self.error_raise = self.global_params.error_raise

        if self.verbose >= 3:
            self.logger.info(f"Cross-validating {method_name}")

        self.global_parameters = global_parameters

        self.ml_grid_object_iter = ml_grid_object

        self.X_train = self.ml_grid_object_iter.X_train
        self.y_train = self.ml_grid_object_iter.y_train
        self.X_test = self.ml_grid_object_iter.X_test
        self.y_test = self.ml_grid_object_iter.y_test
        self.X_test_orig = self.ml_grid_object_iter.X_test_orig
        self.y_test_orig = self.ml_grid_object_iter.y_test_orig

        # Data is assumed to be NumPy arrays from the time-series pipeline.
        # No pandas-specific conversions are needed.

        max_param_space_iter_value = self.global_params.max_param_space_iter_value

        if (
            self.ml_grid_object_iter.local_param_dict.get("max_param_space_iter_value")
            is not None
        ):
            max_param_space_iter_value = self.ml_grid_object_iter.local_param_dict.get(
                "max_param_space_iter_value"
            )

        # Optimize y_test and y_test_orig to reduce metric calculation overhead
        self.y_test = self._optimize_y(self.y_test)
        self.y_test_orig = self._optimize_y(self.y_test_orig)

        # Use faster CV strategy in test mode
        if getattr(self.global_parameters, "test_mode", False):
            self.logger.info("Test mode enabled. Using fast KFold(n_splits=2) for CV.")
            self.cv = KFold(n_splits=2, shuffle=True, random_state=1)
        else:
            self.cv = RepeatedKFold(
                n_splits=2,
                n_repeats=2,
                random_state=1,
            )

        start = time.time()

        current_algorithm = algorithm_implementation

        # --- ML_GRID_FIX: Redirect aeon model saving ---
        # If the model is a deep learning model, set its save path to the experiment directory.
        if BaseDeepClassifier and isinstance(current_algorithm, BaseDeepClassifier):
            if (
                self.project_score_save_class_instance
                and self.project_score_save_class_instance.experiment_dir
            ):
                save_path = self.project_score_save_class_instance.experiment_dir

                # Ensure path ends with separator because aeon uses string concatenation
                # for file paths (e.g. self.file_path + self.init_file_name)
                if not str(save_path).endswith(os.sep):
                    save_path = str(save_path) + os.sep

                os.makedirs(save_path, exist_ok=True)
                self.logger.info(f"Redirecting aeon model saves to: {save_path}")

                # Set the file_path for saving models.
                # Also, explicitly enable saving the best model as a sensible default.
                # The user can override this via the parameter_space if needed.
                params_to_set = {
                    "file_path": str(save_path),
                    "save_best_model": True,
                    "save_last_model": False,
                    "save_init_model": False,
                }
                try:
                    current_algorithm.set_params(**params_to_set)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Could not set save parameters on {method_name}: {e}"
                    )

        if hasattr(current_algorithm, "set_params"):
            # Check for deep learning models that might be verbose
            dl_names = [
                "keras",
                "neural",
                "fcn",
                "tapnet",
                "encoder",
                "resnet",
                "inception",
                "cnn",
                "mlp",
            ]
            if any(name in method_name.lower() for name in dl_names):
                ml_grid_object.logger.info("Silencing Keras verbose output.")
                try:
                    # Try setting verbose directly (for non-pipelines)
                    current_algorithm.set_params(verbose=0)
                except Exception:
                    pass
                try:
                    # Try setting model__verbose (for pipelines)
                    current_algorithm.set_params(model__verbose=0)
                except Exception:
                    pass

        if torch and "simbsig" in str(type(algorithm_implementation)):
            if not torch.cuda.is_available():
                self.logger.info(
                    "No CUDA GPU detected. Forcing simbsig model to use CPU."
                )
                if hasattr(current_algorithm, "set_params"):
                    current_algorithm.set_params(device="cpu")
            else:
                self.logger.info(
                    "CUDA GPU detected. Allowing simbsig model to use GPU."
                )

        self.logger.debug(f"Algorithm implementation: {algorithm_implementation}")

        parameters = parameter_space

        if ml_grid_object.verbose >= 3:
            self.logger.debug(
                f"algorithm_implementation: {algorithm_implementation}, type: {type(algorithm_implementation)}"
            )

        if self.global_params.bayessearch:
            self.logger.debug("Validating parameter space for Bayesian search...")

            def _unwrap_skopt_lists(space):
                """
                Recursively unwraps lists containing a single item. This is crucial for
                BayesSearchCV, which misinterprets a single-item list (e.g., `[[1, 2]]`)
                as a Categorical dimension with one unhashable category (`[1, 2]`),
                leading to a TypeError. This function unwraps it to just `[1, 2]`.
                """
                if isinstance(space, dict):
                    return {k: _unwrap_skopt_lists(v) for k, v in space.items()}
                elif isinstance(space, list):
                    # If it's a list of dicts (e.g. for multiple architectures), recurse
                    if len(space) > 0 and isinstance(space[0], dict):
                        return [_unwrap_skopt_lists(item) for item in space]
                    # If it's a single-item list, unwrap it and recurse.
                    if len(space) == 1:
                        self.logger.info(f"Auto-unwrapping single-item list: {space}")
                        return _unwrap_skopt_lists(space[0])
                    return space
                else:
                    return space

            parameter_space = _unwrap_skopt_lists(parameter_space)

            def _validate_and_correct_bayes_space(space):
                """
                Recursively validates and corrects a parameter space for BayesSearchCV.
                - Wraps scalar values (e.g., 1234) in a list ([1234]) to mark them
                  as fixed parameters for skopt.
                """
                if isinstance(space, list):
                    # This handles lists of dictionaries (e.g., for different model architectures)
                    return [_validate_and_correct_bayes_space(s) for s in space]

                if not isinstance(space, dict):
                    return space

                corrected_space = {}
                for key, value in space.items():
                    if is_skopt_space(value) or isinstance(
                        value, (list, tuple, np.ndarray)
                    ):
                        # This is already a skopt dimension or a list of choices, which is valid.
                        corrected_space[key] = value
                    else:
                        # This is a scalar value (e.g., int, float, str). Skopt requires
                        # fixed parameters to be in a single-element list.
                        self.logger.info(
                            f"Auto-correcting param '{key}' for BayesSearch: wrapping scalar in a list."
                        )
                        corrected_space[key] = [value]
                return corrected_space

            parameter_space = _validate_and_correct_bayes_space(parameter_space)

        # Always validate the parameter space to filter out invalid keys.
        # This is crucial for both grid search and Bayesian search.
        parameters = validate_parameters_helper(
            algorithm_implementation=algorithm_implementation,
            parameters=parameter_space,
            ml_grid_object=ml_grid_object,
        )

        try:
            n_iter_v = getattr(self.global_params, "n_iter", 2)
            if n_iter_v is None:
                n_iter_v = 2
            n_iter_v = int(n_iter_v)
        except (ValueError, TypeError):
            self.logger.warning(
                "Invalid or missing n_iter in global_params. Defaulting to 2."
            )
            n_iter_v = 2

        local_n_iter = self.ml_grid_object_iter.local_param_dict.get("n_iter")
        if local_n_iter is not None:
            try:
                n_iter_v = int(local_n_iter)
                self.logger.info(
                    f"Overriding global n_iter with local value: {n_iter_v}"
                )
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid local n_iter value: {local_n_iter}. Ignoring override."
                )

        if max_param_space_iter_value is not None:
            if n_iter_v > max_param_space_iter_value:
                self.logger.info(
                    f"Capping n_iter ({n_iter_v}) to max_param_space_iter_value ({max_param_space_iter_value})"
                )
                n_iter_v = max_param_space_iter_value

        is_bayes_space = False
        if isinstance(parameter_space, list):
            for space in parameter_space:
                if isinstance(space, dict) and any(
                    is_skopt_space(v) for v in space.values()
                ):
                    is_bayes_space = True
                    break
        elif isinstance(parameter_space, dict):
            if any(is_skopt_space(v) for v in parameter_space.values()):
                is_bayes_space = True

        if (
            not self.global_params.bayessearch
            and not random_grid_search
            and not is_bayes_space
        ):
            try:
                pg = len(ParameterGrid(parameter_space))
                self.logger.info(f"Parameter grid size: {pg}")
            except TypeError:
                self.logger.warning(
                    "Could not calculate ParameterGrid size (likely skopt objects)."
                )
                pg = "N/A"
        else:
            self.logger.info(f"Using n_iter={n_iter_v} for search.")
            pg = "N/A"

        if "kneighbors" in method_name.lower() or "simbsig" in method_name.lower():
            self._adjust_knn_parameters(parameter_space)
            self.logger.debug(
                "Adjusted KNN n_neighbors parameter space to prevent errors on small CV folds."
            )

        original_grid_n_jobs = self.global_parameters.grid_n_jobs
        if is_gpu_model:
            self.global_parameters.grid_n_jobs = 1

        try:
            search = HyperparameterSearch(
                algorithm=current_algorithm,
                parameter_space=parameters,
                method_name=method_name,
                global_params=self.global_parameters,
                sub_sample_pct=self.sub_sample_param_space_pct,
                max_iter=n_iter_v,
                ml_grid_object=ml_grid_object,
                cv=self.cv,
            )

            if self.global_parameters.verbose >= 3:
                self.logger.debug("Running hyperparameter search")

            default_scores = {
                "test_accuracy": np.array([0.5]),
                "test_f1": np.array([0.5]),
                "test_auc": np.array([0.5]),
                "fit_time": np.array([0]),
                "score_time": np.array([0]),
                "train_score": np.array([0.5]),
                "test_recall": np.array([0.5]),
            }

            failed = False
            scores = None
            start_time = time.time()

            try:
                # Data is already in NumPy format for time-series models
                X_train_search = self.X_train
                y_train_search = self._optimize_y(self.y_train)

                with sklearn.config_context(skip_parameter_validation=True):
                    with joblib.parallel_backend("threading"):
                        current_algorithm = search.run_search(
                            X_train_search, y_train_search
                        )

            except TimeoutError:
                self.logger.warning("Timeout occurred during hyperparameter search.")
                failed = "Timeout"
                scores = default_scores

            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt during hyperparameter search.")
                failed = "KeyboardInterrupt"
                scores = default_scores

            except Exception as e:
                self.logger.error(
                    f"An exception occurred during hyperparameter search for {method_name}: {e}",
                    exc_info=True,
                )
                raise e
        finally:
            self.global_parameters.grid_n_jobs = original_grid_n_jobs

        if not failed and self.global_parameters.verbose >= 3:
            self.logger.debug("Fitting final model")

        if not failed and len(np.unique(self.y_train)) < 2:
            raise ValueError(
                "Only one class present in y_train. ROC AUC score is not defined "
                "in that case. grid_search_cross_validate>>>cross_validate"
            )

        if not failed and self.global_parameters.verbose >= 1:
            self.logger.info("Getting cross validation scores")
            self.logger.debug(
                f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}"
            )
            self.logger.debug(
                f"y_train value counts:\n{pd.Series(self.y_train).value_counts()}"
            )

        time_threshold = 60

        is_keras_model = isinstance(
            current_algorithm, (KerasClassifier, KerasClassifierClass)
        )

        final_cv_n_jobs = 1 if is_gpu_model or is_keras_model else grid_n_jobs
        if final_cv_n_jobs == 1:
            self.logger.debug(
                "GPU or Keras model detected. Forcing n_jobs=1 for final cross-validation."
            )

        try:
            if failed:
                raise TimeoutError

            # For time-series, data is already in the correct NumPy format
            X_train_final = self.X_train
            y_train_final = self._optimize_y(self.y_train)

            scores = None

            force_second_cv = self.ml_grid_object_iter.local_param_dict.get(
                "force_second_cv", getattr(self.global_params, "force_second_cv", False)
            )

            if force_second_cv:
                self.logger.info(
                    "force_second_cv is True. Skipping cached result extraction to run fresh cross-validation."
                )

            if (
                not force_second_cv
                and hasattr(current_algorithm, "cv_results_")
                and hasattr(current_algorithm, "best_index_")
            ):
                try:
                    self.logger.info(
                        "Using cached cross-validation results from HyperparameterSearch."
                    )
                    results = current_algorithm.cv_results_
                    index = current_algorithm.best_index_
                    n_splits = self.cv.get_n_splits()

                    temp_scores = {}
                    if "split0_fit_time" in results:
                        temp_scores["fit_time"] = np.array(
                            [
                                results[f"split{k}_fit_time"][index]
                                for k in range(n_splits)
                            ]
                        )
                    else:
                        temp_scores["fit_time"] = np.full(
                            n_splits, results["mean_fit_time"][index]
                        )

                    if "split0_score_time" in results:
                        temp_scores["score_time"] = np.array(
                            [
                                results[f"split{k}_score_time"][index]
                                for k in range(n_splits)
                            ]
                        )
                    else:
                        default_times = np.zeros(index + 1)
                        temp_scores["score_time"] = np.full(
                            n_splits,
                            results.get("mean_score_time", default_times)[index],
                        )

                    for metric in self.metric_list:
                        test_key = f"test_{metric}"
                        temp_scores[test_key] = np.array(
                            [
                                results[f"split{k}_test_{metric}"][index]
                                for k in range(n_splits)
                            ]
                        )
                        train_key = f"train_{metric}"
                        train_col = f"split0_train_{metric}"
                        if train_col in results:
                            temp_scores[train_key] = np.array(
                                [
                                    results[f"split{k}_train_{metric}"][index]
                                    for k in range(n_splits)
                                ]
                            )
                    scores = temp_scores
                except Exception as e:
                    self.logger.warning(
                        f"Could not extract cached CV results: {e}. Falling back to standard CV."
                    )
                    scores = None

            if scores is None:
                if isinstance(
                    current_algorithm, (KerasClassifier, KerasClassifierClass)
                ):
                    self.logger.debug("Fitting Keras model with internal CV handling.")
                    current_algorithm.fit(
                        self.X_train, self.y_train, cv=self.cv, verbose=0
                    )
                    scores = {
                        "test_roc_auc": [
                            current_algorithm.score(self.X_test, self.y_test)
                        ]
                    }
                else:
                    with sklearn.config_context(skip_parameter_validation=True):
                        backend = "threading"
                        with joblib.parallel_backend(backend):
                            scores = cross_validate(
                                current_algorithm,
                                X_train_final,
                                y_train_final,
                                scoring=self.metric_list,
                                cv=self.cv,
                                n_jobs=final_cv_n_jobs,
                                pre_dispatch="2*n_jobs",
                                error_score=self.error_raise,
                            )

                    if isinstance(
                        current_algorithm, (KerasClassifier, KerasClassifierClass)
                    ):
                        try:
                            self.logger.debug(
                                "Pre-compiling TensorFlow predict function to avoid retracing."
                            )
                            n_features = self.X_train.shape[2]  # For 3D TS data
                            input_signature = [
                                tf.TensorSpec(
                                    shape=(None, None, n_features), dtype=tf.float32
                                )
                            ]
                            current_algorithm.model_.predict.get_concrete_function(
                                input_signature
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Could not pre-compile TF function. Performance may be impacted. Error: {e}"
                            )

        except ValueError as e:
            # This can happen with some estimators on certain data splits
            self.logger.error(
                f"An unexpected ValueError occurred during cross-validation: {e}",
                exc_info=True,
            )
            failed = True
            scores = default_scores

        except RuntimeError as e:
            self.logger.error(
                f"A RuntimeError occurred during cross-validation: {e}",
                exc_info=True,
            )
            self.logger.warning("Returning default scores.")
            failed = True
            scores = default_scores

        except TimeoutError:
            self.logger.warning("Timeout occurred during cross-validation.")
            failed = "Timeout"
            scores = default_scores

        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt during cross-validation.")
            failed = "KeyboardInterrupt"
            scores = default_scores

        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during cross-validation: {e}",
                exc_info=True,
            )
            failed = True
            scores = default_scores

        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.global_parameters.verbose >= 1:
            if elapsed_time > time_threshold:
                self.logger.warning(
                    f"Cross-validation took too long ({elapsed_time:.2f} seconds). "
                    "Consider optimizing the parameters or reducing CV folds."
                )
            else:
                self.logger.info(
                    f"Cross-validation for {method_name} completed in {elapsed_time:.2f} seconds."
                )

        if self.global_parameters.verbose >= 4:
            debug_print_statements_class(scores).debug_print_scores()

        try:
            best_pred_orig = current_algorithm.predict(self.X_test)
        except Exception:
            best_pred_orig = np.zeros(len(self.X_test))

        if self.project_score_save_class_instance:
            self.project_score_save_class_instance.update_score_log(
                ml_grid_object=ml_grid_object,
                scores=scores,
                best_pred_orig=best_pred_orig,
                current_algorithm=current_algorithm,
                method_name=method_name,
                pg=pg,
                start=start,
                n_iter_v=n_iter_v,
                failed=failed,
            )
        else:
            self.logger.warning(
                "No project_score_save_class_instance provided. Skipping score logging."
            )

        try:
            y_test_np = self.y_test
            auc = metrics.roc_auc_score(y_test_np, best_pred_orig)
        except Exception:
            auc = 0.5

        self.grid_search_cross_validate_score_result = auc

    def _optimize_y(self, y):
        """Helper to optimize y for sklearn to reduce type_of_target overhead."""
        if hasattr(y, "dtype") and isinstance(y.dtype, pd.CategoricalDtype):
            y_opt = y.cat.codes.values
        elif hasattr(y, "values"):
            y_opt = y.values
        else:
            y_opt = y

        if not pd.api.types.is_integer_dtype(y_opt):
            try:
                y_opt = y_opt.astype(int)
            except (ValueError, TypeError):
                y_opt, _ = pd.factorize(y_opt, sort=True)
                y_opt = y_opt.astype(int)

        return np.ascontiguousarray(y_opt)

    def _adjust_knn_parameters(self, parameter_space: Union[Dict, List[Dict]]):
        """
        Dynamically adjusts the 'n_neighbors' parameter for KNN-based models
        to prevent errors on small datasets during cross-validation.
        """
        self.cv.get_n_splits()

        dummy_indices = np.arange(len(self.X_train))
        train_indices, _ = next(self.cv.split(dummy_indices))
        n_samples_train_fold = len(train_indices)
        n_samples_test_fold = len(self.X_train) - n_samples_train_fold
        max_n_neighbors = max(1, n_samples_train_fold)

        self.logger.debug(
            f"KNN constraints - train_fold_size={n_samples_train_fold}, "
            f"test_fold_size={n_samples_test_fold}, max_n_neighbors={max_n_neighbors}"
        )

        def adjust_param(param_value):
            if is_skopt_space(param_value):
                # Handle Integer and Real spaces which have 'high' and 'low'
                if hasattr(param_value, "high"):
                    new_high = min(param_value.high, max_n_neighbors)
                    new_low = min(param_value.low, new_high)
                    param_value.high = new_high
                    param_value.low = new_low
                    self.logger.debug(
                        f"Adjusted skopt Integer/Real space: low={new_low}, high={new_high}"
                    )
                    return param_value
                # Handle Categorical spaces which have 'categories'
                elif hasattr(param_value, "categories"):
                    new_categories = [
                        cat for cat in param_value.categories if cat <= max_n_neighbors
                    ]
                    if not new_categories:
                        self.logger.warning(
                            f"All n_neighbors categories filtered out. Using [{max_n_neighbors}]"
                        )
                        new_categories = [max_n_neighbors]
                    # Create a new Categorical object as its properties are read-only
                    self.logger.debug(
                        f"Filtered skopt Categorical space to: {new_categories}"
                    )
                    return Categorical(new_categories)

            elif isinstance(param_value, (list, np.ndarray)):
                new_param_value = [n for n in param_value if n <= max_n_neighbors]
                if not new_param_value:
                    self.logger.warning(
                        f"All n_neighbors values filtered out. Using [{max_n_neighbors}]"
                    )
                    return [max_n_neighbors]
                self.logger.debug(f"Filtered n_neighbors list: {new_param_value}")
                return new_param_value
            return param_value

        if isinstance(parameter_space, list):
            for params in parameter_space:
                if "n_neighbors" in params:
                    params["n_neighbors"] = adjust_param(params["n_neighbors"])
        elif isinstance(parameter_space, dict) and "n_neighbors" in parameter_space:
            parameter_space["n_neighbors"] = adjust_param(
                parameter_space["n_neighbors"]
            )
