"""AutoGluon Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for AutoGluon's TabularPredictor.
"""

import logging
import os
import shutil
import tempfile
import uuid
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Attempt to import AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    from autogluon.core.utils.exceptions import TimeLimitExceeded
    from ml_grid.util.global_params import global_parameters
except ImportError:
    TabularPredictor = None
    TimeLimitExceeded = TimeoutError

    # Mock object to avoid errors if autogluon is not installed
    class MockGlobalParams:
        pass

    global_parameters = MockGlobalParams()

logger = logging.getLogger(__name__)


class AutoGluonClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for AutoGluon TabularPredictor."""

    def __init__(
        self,
        time_limit: int = 120,
        presets: Optional[str] = None,
        eval_metric: str = "accuracy",
        problem_type: Optional[str] = None,
        seed: int = 42,
        verbosity: int = 2,
        path: Optional[str] = None,
        excluded_model_types: Optional[List[str]] = None,
        hyperparameters: Optional[dict] = None,
    ):
        self.time_limit = time_limit
        self.presets = presets
        self.eval_metric = eval_metric
        self.problem_type = problem_type
        self.seed = seed
        self.verbosity = verbosity
        self.path = path
        self.excluded_model_types = excluded_model_types
        self.hyperparameters = hyperparameters

        self.predictor_ = None
        self.classes_ = None
        self._temp_dir = None
        self.model_id = None  # For compatibility with internal logging if needed
        self.timed_out_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoGluonClassifier":
        if TabularPredictor is None:
            raise ImportError(
                "AutoGluon is not installed. Please install it to use AutoGluonClassifier."
            )

        # Validate input X
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]

        # Validate input y
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")

        # Ensure y has a name
        if y.name is None:
            y.name = "target"

        label_column = y.name

        # Prepare training data
        train_data = X.copy()
        train_data[label_column] = y.values

        effective_time_limit = self.time_limit

        # Handle path
        if self.path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="autogluon_")
            # AutoGluon warns if the directory exists. Since mkdtemp creates it,
            # we remove it so AutoGluon can recreate it without warning.
            shutil.rmtree(self._temp_dir)
            model_path = self._temp_dir
        else:
            model_path = self.path

        # Check for FastAI and exclude if not installed to prevent ImportErrors
        excluded_models = (
            self.excluded_model_types if self.excluded_model_types is not None else []
        )
        try:
            import fastai  # noqa: F401, E402
        except ImportError:
            if "FASTAI" not in excluded_models:
                excluded_models = list(excluded_models) + ["FASTAI"]

        # Exclude NeuralNetTorch (NN_TORCH) by default for stability in unit tests, as it can be
        # resource-intensive and prone to filesystem errors with Ray's checkpointing.
        if "NN_TORCH" not in excluded_models:
            excluded_models.append("NN_TORCH")

        # Initialize predictor
        self.predictor_ = TabularPredictor(
            label=label_column,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=model_path,
            verbosity=self.verbosity,
        )

        # The seed for AutoGluon's HPO search should be passed in hyperparameter_tune_kwargs.
        # This ensures reproducibility of the internal model selection and tuning process.
        hyperparameter_tune_kwargs = {
            "searcher": "random",  # Default searcher
            "scheduler": "local",  # Default scheduler
            "searcher_options": {"seed": self.seed},
        }

        # Apply a safety buffer to the time limit to ensure we return before any external timeout.
        # AutoGluon attempts to stop training by the limit, but saving/cleanup adds overhead.
        safe_time_limit = effective_time_limit
        if effective_time_limit and effective_time_limit > 20:
            # Reserve 10% for overhead, with a floor of 15s and a ceiling of 60s.
            buffer = min(60, max(15, int(effective_time_limit * 0.10)))
            safe_time_limit = max(effective_time_limit - buffer, 10)
            logger.info(
                f"Reduced AutoGluon time_limit from {effective_time_limit}s to {safe_time_limit}s to allow for overhead."
            )

        # Set up arguments for AutoGluon's fit method
        fit_args = kwargs.copy()
        fit_args.update(
            {
                "time_limit": safe_time_limit,
                "hyperparameter_tune_kwargs": hyperparameter_tune_kwargs,
                "excluded_model_types": excluded_models,
                "dynamic_stacking": False,
            }
        )

        # Prioritize hyperparameters, then presets. If neither, use a fast default for tests.
        if self.hyperparameters:
            fit_args["hyperparameters"] = self.hyperparameters
        elif self.presets:
            fit_args["presets"] = self.presets
        else:
            logger.info(
                "No presets or hyperparameters specified. Using fast default for unit testing: {'GBM': {}}"
            )
            fit_args["hyperparameters"] = {"GBM": {}}

        # Log configuration to assist with debugging silent/long runs
        logger.info(f"Starting AutoGluon fit. Path: {model_path}")
        logger.info(
            f"Time limit: {safe_time_limit}s (Effective: {effective_time_limit}s)"
        )
        logger.info(f"Verbosity: {self.verbosity}")

        if fit_args.get("presets"):
            logger.info(f"Presets: {fit_args['presets']}")

        if fit_args.get("hyperparameters"):
            # Log keys only to avoid flooding logs if hyperparameters are large
            logger.info(
                f"Hyperparameters keys: {list(fit_args['hyperparameters'].keys()) if isinstance(fit_args['hyperparameters'], dict) else 'custom'}"
            )

        # Mitigate nested parallelism when running inside a joblib worker.
        # If the JOBLIB_SPAWNED_PROCESS env var is present, we are in a worker.
        # Constraining num_cpus prevents resource over-subscription.
        if "JOBLIB_SPAWNED_PROCESS" in os.environ:
            logger.info(
                "Detected execution within a joblib worker. Constraining AutoGluon to use 1 CPU core."
            )
            if self.verbosity > 0:
                logger.warning(
                    "Running inside joblib worker. AutoGluon output may be captured/suppressed by the parent process."
                )
            fit_args["num_cpus"] = 1

        # Fit predictor
        try:
            self.predictor_.fit(train_data, **fit_args)
        except TimeLimitExceeded:
            self.timed_out_ = True
            logger.warning(
                "AutoGluon TimeLimitExceeded during fit. Checking if any models were trained..."
            )
            if self.predictor_.model_names():
                logger.info(
                    "At least one model was trained. Continuing with partial fit."
                )
            else:
                raise
        except Exception as e:
            logger.error(f"AutoGluon fit failed with error: {e}")
            raise

        # Check if any models were actually trained
        if not self.predictor_.model_names():
            msg = "AutoGluon failed to train any models."
            logger.error(msg)
            raise RuntimeError(msg)

        self.classes_ = np.array(self.predictor_.class_labels)
        self.model_id = f"autogluon_{uuid.uuid4().hex}"

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]

        return self.predictor_.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "classes_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"feature_{i}" for i in range(X.shape[1])]

        # AutoGluon returns a DataFrame with class labels as columns
        probas_df = self.predictor_.predict_proba(X)

        # Ensure we return columns in the same order as self.classes_
        if self.classes_ is not None:
            return probas_df[self.classes_].values

        return probas_df.values

    def __del__(self):
        # Cleanup temporary directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
