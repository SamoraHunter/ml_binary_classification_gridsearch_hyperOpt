"""Defines the TabPFN Classifier model class."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from skopt.space import Categorical, Integer

from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.getLogger("ml_grid").warning(
        "TabPFN not available. Install with: pip install tabpfn"
    )

logging.getLogger("ml_grid").debug("Imported TabPFNClassifier class")


class TabPFNClassifierClass(BaseEstimator, ClassifierMixin):
    """TabPFN Classifier with support for hyperparameter tuning.

    TabPFN is a foundation model for tabular data that performs well on small
    to medium-sized datasets (up to 50,000 rows). It requires GPU for optimal
    performance on datasets larger than ~1000 samples.

    Note: TabPFN-2.5 model weights require accepting license terms at:
    https://huggingface.co/Prior-Labs/tabpfn_2_5
    """

    def __init__(
        self,
        parameter_space_size: Optional[str] = None,
        # Hyperparameters for scikit-learn compatibility
        model_version: str = "v2.5_default",
        device: str = "cpu",
        n_estimators: int = 4,
        subsample_samples: Optional[int] = None,
        random_state: int = 42,
    ):
        """Initializes the TabPFNClassifierClass.

        Args:
            parameter_space_size (Optional[str]): Size of the parameter space for
                optimization. Defaults to None.

            model_version (str): The version of the TabPFN model to use.
            device (str): The device to run the model on ('cpu' or 'cuda').
            n_estimators (int): Number of ensemble members.
            subsample_samples (Optional[int]): Subsample size for large datasets.
            random_state (int): Random state for reproducibility.
        Raises:
            ImportError: If TabPFN is not installed.
        """
        if not TABPFN_AVAILABLE:
            raise ImportError(
                "TabPFN is not installed. Install with: pip install tabpfn"
            )

        # Store scikit-learn hyperparameters
        self.model_version = model_version
        self.device = device
        self.n_estimators = n_estimators
        self.subsample_samples = subsample_samples
        self.random_state = random_state

        global_params = global_parameters
        self.parameter_space_size = parameter_space_size

        self.algorithm_implementation = self  # The instance itself is the estimator
        self.method_name: str = "TabPFNClassifier"

        self.parameter_vector_space: param_space.ParamSpace = param_space.ParamSpace(
            parameter_space_size
        )
        self.parameter_space: Dict[str, Any]

        if global_params.bayessearch:
            self.parameter_space = {
                # Model version selection
                "model_version": Categorical(
                    [
                        "v2.5_default",  # Default: finetuned on real data
                        "v2.5_synthetic",  # Trained on synthetic data only
                        "v2",  # TabPFN v2
                    ]
                ),
                # Device selection - can be optimized based on availability
                "device": Categorical(["cuda", "cpu"]),
                # Number of ensemble members (more = better but slower)
                "n_estimators": Integer(1, 8),
                # Training subsample size (for large datasets)
                "subsample_samples": Categorical([None, 5000, 10000, 20000]),
                # Random state for reproducibility
                "random_state": Categorical([42]),
            }

        else:
            self.parameter_space = {
                "model_version": ["v2.5_default", "v2.5_synthetic", "v2"],
                "device": ["cuda", "cpu"],
                "n_estimators": [1, 2, 4, 8],
                "subsample_samples": [None, 5000, 10000, 20000],
                "random_state": [42],
            }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the TabPFN model.

        This method uses the hyperparameters set on the instance to create
        and fit the underlying TabPFNClassifier.
        """
        # Apply subsampling if configured
        if self.subsample_samples is not None and len(X) > self.subsample_samples:
            # Use numpy for stable random sampling
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(len(X), self.subsample_samples, replace=False)

            # Handle DataFrame/Series or numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.iloc[indices]
            else:
                X = X[indices]

            if isinstance(y, pd.Series):
                y = y.iloc[indices]
            else:
                y = y[indices]

        # Get the hyperparameters from the instance itself
        params = self.get_params()

        # Check for GPU availability and fallback if necessary
        if params.get("device") == "cuda" and not torch.cuda.is_available():
            logging.getLogger("ml_grid").warning(
                "TabPFN device set to 'cuda' but no CUDA GPU found. Falling back to 'cpu'."
            )
            params["device"] = "cpu"

        # This logic was originally in create_model
        model_version = params.pop("model_version", "v2.5_default")

        # Filter out non-TabPFN params that might be in get_params()
        valid_tabpfn_params = ["device", "n_estimators", "random_state"]
        params_copy = {k: v for k, v in params.items() if k in valid_tabpfn_params}

        if model_version == "v2.5_synthetic":
            params_copy["model_path"] = "tabpfn-v2.5-classifier-v2.5_default-2.ckpt"

        if model_version == "v2":
            self._estimator = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, **params_copy
            )
        else:
            self._estimator = TabPFNClassifier(**params_copy)

        self._estimator.fit(X, y)
        self.classes_ = self._estimator.classes_
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Makes predictions using the fitted model."""
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns probability estimates for predictions."""
        return self._estimator.predict_proba(X)
