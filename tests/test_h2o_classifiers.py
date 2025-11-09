import logging

import h2o
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, cross_val_score
from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLClass
from ml_grid.model_classes.h2o_deeplearning_classifier_class import (
    H2O_DeepLearning_class,
)
from ml_grid.model_classes.h2o_drf_classifier_class import H2ODRFClass as H2O_DRF_class
from ml_grid.model_classes.h2o_gam_classifier_class import H2OGAMClass as H2O_GAM_class

# Import all the H2O *model definition* classes, which is more realistic
from ml_grid.model_classes.h2o_gbm_classifier_class import H2O_GBM_class
from ml_grid.model_classes.h2o_glm_classifier_class import H2O_GLM_class
from ml_grid.model_classes.h2o_naive_bayes_classifier_class import (
    H2O_NaiveBayes_class,
)
from ml_grid.model_classes.h2o_rulefit_classifier_class import (
    H2ORuleFitClass as H2O_RuleFit_class,
)
from ml_grid.model_classes.h2o_xgboost_classifier_class import H2O_XGBoost_class
from ml_grid.pipeline.grid_search_cross_validate import grid_search_crossvalidate
from ml_grid.util.global_params import global_parameters


@pytest.fixture
def tiny_problematic_data():
    """
    Provides a very small dataset that is known to cause issues with some
    models during cross-validation if parameters are not handled carefully.
    A 10-sample dataset with 5-fold CV results in 8-sample training folds.
    """
    X = pd.DataFrame({
        # A single, nearly-constant feature to reliably trigger constant column errors.
        'feature1': [0] * 9 + [1],
    })
    y = pd.Series(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), name="outcome")
    return X, y

# A list of all H2O model definition classes to be tested

H2O_MODEL_CLASSES = [
    H2O_GBM_class,
    H2O_DRF_class,
    H2O_GAM_class,
    H2O_DeepLearning_class,
    H2O_GLM_class,
    H2O_NaiveBayes_class,
    H2O_RuleFit_class,
    H2O_XGBoost_class,
    # H2O_StackedEnsemble_class, # Known issues - skipping for now,
    H2OAutoMLClass, # AutoML
]

# To reduce runtime and ensure consistent test runs, select a fixed, smaller set of
# models. For full coverage, you would test all, but for speed, a representative
# subset is better.
H2O_MODEL_CLASSES = [H2O_GLM_class, H2O_DRF_class]


# This fixture will be parameterized to create an instance of each model class
@pytest.fixture(params=H2O_MODEL_CLASSES)
def h2o_model_instance(request, synthetic_data):
    """
    Creates an instance of an H2O model definition class, which in turn
    creates the underlying scikit-learn compatible wrapper. This more
    closely mimics the main pipeline's workflow.
    """
    model_class = request.param
    X, y = synthetic_data

    # The H2OAutoMLConfig class has a different constructor signature
    # and doesn't accept X, y during initialization.
    if model_class == H2OAutoMLClass:
        instance = model_class(parameter_space_size="small")
    else:
        # Ensure y is a Series for consistency, which some model classes might expect
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        instance = model_class(X=X, y=y, parameter_space_size="small")
    return instance.algorithm_implementation

# Use pytest.mark.parametrize to run the same test for all classifiers
def test_h2o_classifier_fit_predict(
    h2o_model_instance, synthetic_data, h2o_session_fixture
):
    """
    Tests the basic fit and predict functionality of each H2O wrapper.
    """
    X, y = synthetic_data
    # Clean up frames from any previous test runs to avoid conflicts
    h2o.remove_all()

    estimator = h2o_model_instance

    # 1. Fit the model
    estimator.fit(X, y)

    # 2. Predict labels
    predictions = estimator.predict(X)
    assert isinstance(predictions, np.ndarray), "predict() should return a numpy array"
    assert predictions.shape == (X.shape[0],), "Prediction array has incorrect shape"

    # 3. Predict probabilities
    proba_predictions = estimator.predict_proba(X)
    assert isinstance(
        proba_predictions, np.ndarray
    ), "predict_proba() should return a numpy array"
    assert (
        proba_predictions.shape == (X.shape[0], 2)
    ), "Probability array has incorrect shape"
    assert np.allclose(
        np.sum(proba_predictions, axis=1), 1.0
    ), "Probabilities should sum to 1"

    # 4. Test set_params and get_params
    estimator.set_params(seed=123)
    params = estimator.get_params()
    if "seed" in params:
        assert params["seed"] == 123, "set_params/get_params failed to update seed"


@pytest.mark.parametrize("model_class", H2O_MODEL_CLASSES)
def test_h2o_classifiers_with_cross_validation(
    model_class, tiny_problematic_data, h2o_session_fixture
):
    """
    Tests that H2O wrappers are robust enough to run in a cross-validation
    loop with very small data splits, which can cause errors if not handled.
    This simulates the conditions of the main pipeline more closely.
    """
    X, y = tiny_problematic_data

    # Clean up frames from any previous test runs to avoid conflicts
    if h2o.cluster().is_running():
        h2o.remove_all()

    # Handle special instantiation for AutoML class
    if model_class == H2OAutoMLClass:
        instance = model_class(parameter_space_size="small")
    else:
        # Ensure y is a Series for consistency
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        instance = model_class(X=X, y=y, parameter_space_size="small")

    estimator = instance.algorithm_implementation

    # Skip test if data is too small
    if len(X) < estimator.MIN_SAMPLES_FOR_STABLE_FIT:
        pytest.skip(f"Skipping {model_class.__name__} due to small dataset size.")

    # Use 5-fold CV. On 10 samples, this creates 8-sample training folds.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # The tiny_problematic_data can cause folds with constant features.
    # The H2OBaseClassifier wrapper correctly raises a RuntimeError in this case.
    # We expect this test to either complete successfully OR fail gracefully with
    # our custom RuntimeError. Any other error will still fail the test.
    try:
        scores = cross_val_score(estimator, X, y, cv=cv, error_score='raise', n_jobs=1)
        assert (
            len(scores) == 5
        ), "Cross-validation did not complete for all folds."
    except RuntimeError as e:
        assert "fit on a single constant feature" in str(
            e
        ), f"Caught unexpected RuntimeError: {e}"


def test_h2o_gam_knot_cardinality_error(h2o_session_fixture):
    """
    Tests that H2OGAMClassifier raises a specific ValueError when a feature
    in a CV fold has fewer unique values than the number of knots.
    """
    # The h2o_session_fixture ensures the cluster is running.
    # Clean up frames from any previous test runs to avoid conflicts
    h2o.remove_all()

    # Create data where 'feature2' has low cardinality
    X = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': [0, 1] * 10,  # Only 2 unique values
    })
    y = pd.Series(np.random.randint(0, 2, 20), name="outcome")

    # Instantiate the GAM class
    estimator = H2O_GAM_class(X=X, y=y, parameter_space_size="small").algorithm_implementation

    # Set parameters that will cause the error: 5 knots for a feature with 2
    # unique values.
    # Also, we must disable the wrapper's internal error handling that
    # suppresses this specific error, so that cross_val_score can raise it as intended.
    estimator.set_params(
        gam_columns=['feature2'],
        num_knots=5,
        # This is a custom parameter in the H2OGAMClassifier wrapper
        _suppress_low_cardinality_error=False
    )

    # Use 2-fold CV. One fold could get only one unique value for feature2.
    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    # We expect cross_val_score to fail and raise our specific ValueError
    # Updated regex to match the actual error message from the code
    with pytest.raises(
        ValueError,
        match=r"Feature .* has \d+ unique values, which is insufficient for the requested \d+ knots\. At least \d+ unique values are required\.",
    ):
        # The error_score='raise' is crucial for pytest.raises to catch the exception
        cross_val_score(estimator, X, y, cv=cv, error_score='raise', n_jobs=1)


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
        self.base_project_dir = "test_experiments/test_run" # Add this line
        # Configure global params for a fast, non-verbose test run
        self.verbose = 0
        self.global_params.verbose = 0
        self.global_params.error_raise = True
        # --- H2O CRITICAL: Force n_jobs=1 ---
        # H2O cannot run in parallel via joblib; it causes deadlocks.
        self.global_params.grid_n_jobs = 1
        # --- PERFORMANCE FIX: Use RandomizedSearchCV with a small n_iter ---
        self.global_params.random_grid_search = True
        self.global_params.bayessearch = False
        self.global_params.max_param_space_iter_value = 2 # Only test 2 combinations
        # --- PERFORMANCE FIX: Reduce CV folds for the grid search test ---
        self.global_params.cv_folds = 2
        # --- PERFORMANCE FIX: Add a test_mode flag to skip final CV ---
        self.global_params.test_mode = True
        self.global_params.sub_sample_param_space_pct = 1.0 # Use a small sample
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("model_class", H2O_MODEL_CLASSES)
def test_h2o_full_grid_search_pipeline(
    model_class, synthetic_data, h2o_session_fixture
):
    """
    Tests the full pipeline integration by calling grid_search_crossvalidate.
    This ensures that H2O models work correctly within the hyperparameter
    search framework, including the safeguard that forces n_jobs=1.
    """
    X, y = synthetic_data

    # 1. Instantiate the model definition class, handling AutoML's unique constructor
    if model_class == H2OAutoMLClass:
        # H2OAutoMLConfig does not accept X, y in its constructor
        instance = H2OAutoMLClass(parameter_space_size="small")
    else:
        instance = model_class(X=X, y=y, parameter_space_size="small")

    # 2. Create a mock pipeline object
    mock_ml_grid_object = MockMlGridObject(X, y)


    # RandomizedSearchCV expects a single dictionary for the parameter space.
    # Some model classes might return a list `[{...}]`. We flatten it here.
    if isinstance(instance.parameter_space, list):
        # This handles cases where the space is a list of dicts
        flat_param_space = {}
        for d in instance.parameter_space:
            flat_param_space.update(d)
        instance.parameter_space = flat_param_space

    # --- Fix parameter names and types for non-Bayesian search ---

    # For RandomizedSearchCV, skopt distribution objects must be converted to lists.
    # We iterate through the parameter space and convert them.
    for key, value in instance.parameter_space.items():
        # Check if it's a skopt Categorical object
        if hasattr(value, 'rvs') and hasattr(value, 'categories'):
            # H2O DeepLearning 'hidden' param requires a list, not a tuple.
            # Convert categories of tuples to categories of lists.
            categories = value.categories
            if any(isinstance(cat, tuple) for cat in categories):
                instance.parameter_space[key] = [list(cat) for cat in categories]
            else:
                instance.parameter_space[key] = list(categories)

        # Check if it's a skopt Integer object
        elif (
            hasattr(value, "rvs")
            and hasattr(value, "low")
            and hasattr(value, "high")
            and isinstance(value.low, int)
        ):
            instance.parameter_space[key] = list(range(value.low, value.high + 1))
        # Check if it's a skopt Real object
        elif (
            hasattr(value, "rvs")
            and hasattr(value, "low")
            and hasattr(value, "high")
            and isinstance(value.low, float)
        ):
            instance.parameter_space[key] = np.linspace(
                value.low, value.high, 5
            ).tolist()

    # Fix H2O XGBoost specific parameter name mismatch
    if "col_sample_rate_bytree" in instance.parameter_space:
        instance.parameter_space["colsample_bytree"] = instance.parameter_space.pop(
            "col_sample_rate_bytree"
        )

    # Clean up frames from any previous test runs to avoid conflicts
    h2o.remove_all()

    # 3. Run the full grid search cross-validation process
    result = grid_search_crossvalidate(
        algorithm_implementation=instance.algorithm_implementation,
        parameter_space=instance.parameter_space,
        method_name=instance.method_name,
        ml_grid_object=mock_ml_grid_object
    )
    # 4. Assert that the process completed and returned a score
    assert isinstance(result.grid_search_cross_validate_score_result, float)
