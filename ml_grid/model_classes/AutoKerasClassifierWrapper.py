"""AutoKeras Classifier Wrapper.

This module provides a scikit-learn compatible wrapper for AutoKeras StructuredDataClassifier.
"""

import logging
import os
import shutil
import tempfile
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Attempt to import AutoKeras and TensorFlow
try:
    import autokeras as ak
    import tensorflow as tf
except ImportError:
    ak = None
    tf = None

logger = logging.getLogger(__name__)


class AutoKerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for AutoKeras StructuredDataClassifier."""

    def __init__(
        self,
        max_trials: int = 3,
        epochs: int = 10,
        validation_split: float = 0.2,
        directory: Optional[str] = None,
        seed: int = 42,
        verbose: int = 1,
        overwrite: bool = True,
    ):
        self.max_trials = max_trials
        self.epochs = epochs
        self.validation_split = validation_split
        self.directory = directory
        self.seed = seed
        self.verbose = verbose
        self.overwrite = overwrite

        self.model_ = None
        self._temp_dir = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "AutoKerasClassifierWrapper":
        if ak is None:
            raise ImportError(
                "AutoKeras is not installed. Please install it to use AutoKerasClassifierWrapper."
            )

        # Ensure input is numpy array to avoid AutoKeras ValueError with DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values

        # Handle directory
        if self.directory is None:
            self._temp_dir = tempfile.mkdtemp(prefix="autokeras_")
            dir_path = self._temp_dir
        else:
            dir_path = self.directory

        if tf:
            tf.random.set_seed(self.seed)

        self.model_ = ak.StructuredDataClassifier(
            max_trials=self.max_trials,
            directory=dir_path,
            seed=self.seed,
            overwrite=self.overwrite,
        )

        self.model_.fit(
            x=X,
            y=y,
            epochs=self.epochs,
            validation_split=self.validation_split,
            verbose=self.verbose,
            **kwargs,
        )

        # Check if a model was actually found and can be exported.
        try:
            self.model_.export_model()
        except Exception as e:
            # This typically happens if max_trials is too low and no model is found.
            msg = f"AutoKeras failed to find a usable model (max_trials={self.max_trials}, epochs={self.epochs}). Original error: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        # AutoKeras does not explicitly expose classes_, so we infer them from y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.classes_ = np.unique(y.values)
        else:
            self.classes_ = np.unique(y)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, "model_")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X).flatten()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, "model_")
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Export the underlying Keras model to get probabilities
        keras_model = self.model_.export_model()
        probs = keras_model.predict(X, verbose=0)

        # Handle binary classification case where output is (N, 1)
        if probs.shape[1] == 1:
            return np.hstack([1 - probs, probs])
        return probs

    def __del__(self):
        # Cleanup temporary directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
