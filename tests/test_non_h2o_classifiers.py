# test_non_h2o_classifiers.py

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils.estimator_checks import check_estimator
import logging
from typing import List

from ml_grid.pipeline.grid_search_cross_validate import grid_search_crossvalidate
from ml_grid.util.global_params import global_parameters

# Import a selection of non-H2O model definition classes
from ml_grid.model_classes.adaboost_classifier_class import AdaBoostClassifierClass
from ml_grid.model_classes.gradientboosting_classifier_class import (
    GradientBoostingClassifierClass,
)
from ml_grid.model_classes.logistic_regression_class import LogisticRegressionClass
from ml_grid.model_classes.randomforest_classifier_class import (
    RandomForestClassifierClass,
)
from ml_grid.model_classes.light_gbm_class import LightGBMClassifierWrapper
from ml_grid.model_classes.xgb_classifier_class import XGBClassifierClass


@pytest.fixture
def tiny_problematic_data():
    """
    Provides a very small dataset that can be used to test model robustness
    in cross-validation scenarios with small data splits.
    """
    X = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20),
    })
    y = pd.Series(np.random.randint(0, 2, 20), name="outcome")
    return X, y

# A list of all non-H2O model definition classes to be tested
NON_H2O_MODEL_CLASSES = [
    AdaBoostClassifierClass,
    GradientBoostingClassifierClass,
    LogisticRegressionClass,
    RandomForestClassifierClass,
    LightGBMClassifierWrapper,
    XGBClassifierClass,
]

# This fixture will be parameterized to create an instance of each model class
@pytest.fixture(params=NON_H2O_MODEL_CLASSES)
def model_instance(request, synthetic_data):
    """
    Creates an instance of a model definition class, which in turn
    creates the underlying scikit-learn compatible estimator.
    """
    model_class = request.param
    X, y = synthetic_data
    instance = model_class(X=X, y=y, parameter_space_size="small")
    return instance.algorithm_implementation

# Use pytest.mark.parametrize to run the same test for all classifiers
def test_classifier_fit_predict(model_instance, synthetic_data):
    """
    Tests the basic fit and predict functionality of each classifier.
    """
    X, y = synthetic_data
    estimator = model_instance
    
    # 1. Fit the model
    estimator.fit(X, y)
    
    # 2. Predict labels
    predictions = estimator.predict(X)
    assert isinstance(predictions, np.ndarray), "predict() should return a numpy array"
    assert predictions.shape == (X.shape[0],), "Prediction array has incorrect shape"
    
    # 3. Predict probabilities
    proba_predictions = estimator.predict_proba(X)
    assert isinstance(proba_predictions, np.ndarray), "predict_proba() should return a numpy array"
    assert proba_predictions.shape == (X.shape[0], 2), "Probability array has incorrect shape"
    assert np.allclose(np.sum(proba_predictions, axis=1), 1.0), "Probabilities should sum to 1"

    # 4. Test set_params and get_params
    # Use a parameter that is common to most sklearn estimators
    if 'random_state' in estimator.get_params():
        estimator.set_params(random_state=123)
        params = estimator.get_params()
        assert params['random_state'] == 123, "set_params/get_params failed to update random_state"


@pytest.mark.parametrize("model_class", NON_H2O_MODEL_CLASSES)
def test_classifiers_with_cross_validation(model_class, tiny_problematic_data):
    """
    Tests that classifiers are robust enough to run in a cross-validation
    loop with small data splits.
    """
    X, y = tiny_problematic_data
    instance = model_class(X=X, y=y, parameter_space_size="small")
    estimator = instance.algorithm_implementation
    
    # Use 5-fold CV.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # This test simply checks if cross-validation completes without raising an unhandled error.
    try:
        scores = cross_val_score(estimator, X, y, cv=cv, error_score='raise', n_jobs=1)
        assert len(scores) == 5, "Cross-validation did not complete for all folds."
    except Exception as e:
        pytest.fail(f"{model_class.__name__} failed during cross-validation with small data: {e}")


# A mock class to simulate the main 'pipe' object for integration testing
class MockMlGridObject:
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X_test = X
        self.y_test = y
        self.X_test_orig = X
        self.y_test_orig = y
        self.local_param_dict = {'param_space_size': 'small'}
        self.global_params = global_parameters
        self.base_project_dir = "test_experiments/test_run"
        # Configure global params for a fast, non-verbose test run
        self.verbose = 0
        self.global_params.verbose = 0
        self.global_params.error_raise = True
        # Allow parallel execution for non-H2O models
        self.global_params.grid_n_jobs = 2
        # Use RandomizedSearchCV with a small n_iter for speed
        self.global_params.random_grid_search = True
        self.global_params.bayessearch = False
        self.global_params.max_param_space_iter_value = 2 # Only test 2 combinations
        # Reduce CV folds for the grid search test
        self.global_params.cv_folds = 2
        # Add a test_mode flag to skip final CV
        self.global_params.test_mode = True
        self.global_params.sub_sample_param_space_pct = 1.0
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("model_class", NON_H2O_MODEL_CLASSES)
def test_full_grid_search_pipeline(model_class, synthetic_data):
    """
    Tests the full pipeline integration by calling grid_search_crossvalidate.
    This ensures that models work correctly within the hyperparameter search framework.
    """
    X, y = synthetic_data
    
    # 1. Instantiate the model definition class
    instance = model_class(X=X, y=y, parameter_space_size="small")
    
    # 2. Create a mock pipeline object
    mock_ml_grid_object = MockMlGridObject(X, y)
    
    # --- Fix parameter names and types for non-Bayesian search ---
    
    # For RandomizedSearchCV, skopt distribution objects must be converted to lists or appropriate distributions.
    # We process the parameter space, handling both single dicts and lists of dicts.
    
    param_space = instance.parameter_space
    
    if isinstance(param_space, list):
        processed_param_space = []
        for space_dict in param_space:
            processed_dict = {}
            for key, value in space_dict.items():
                if hasattr(value, 'rvs') and hasattr(value, 'categories'):
                    processed_dict[key] = list(value.categories)
                elif hasattr(value, 'rvs') and hasattr(value, 'low') and hasattr(value, 'high') and isinstance(value.low, int):
                    processed_dict[key] = list(range(value.low, value.high + 1))
                elif hasattr(value, 'rvs') and hasattr(value, 'low') and hasattr(value, 'high') and isinstance(value.low, float):
                    processed_dict[key] = np.linspace(value.low, value.high, 5).tolist()
                else:
                    processed_dict[key] = value
            processed_param_space.append(processed_dict)
        instance.parameter_space = processed_param_space
    elif isinstance(param_space, dict):
        processed_param_space = {}
        for key, value in param_space.items():
            if hasattr(value, 'rvs') and hasattr(value, 'categories'):
                processed_param_space[key] = list(value.categories)
            elif hasattr(value, 'rvs') and hasattr(value, 'low') and hasattr(value, 'high') and isinstance(value.low, int):
                processed_param_space[key] = list(range(value.low, value.high + 1))
            elif hasattr(value, 'rvs') and hasattr(value, 'low') and hasattr(value, 'high') and isinstance(value.low, float):
                processed_param_space[key] = np.linspace(value.low, value.high, 5).tolist()
            else:
                processed_param_space[key] = value
        instance.parameter_space = processed_param_space

    # 3. Run the full grid search cross-validation process
    result = grid_search_crossvalidate(
        algorithm_implementation=instance.algorithm_implementation,
        parameter_space=instance.parameter_space,
        method_name=instance.method_name,
        ml_grid_object=mock_ml_grid_object
    )
    
    # 4. Assert that the process completed and returned a score
    assert isinstance(result.grid_search_cross_validate_score_result, float)
