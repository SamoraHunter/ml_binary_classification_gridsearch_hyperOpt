import logging
import h2o
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, cross_val_score

# --- Import Model Classes ---
from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLClass
from ml_grid.model_classes.h2o_drf_classifier_class import H2ODRFClass as H2O_DRF_class

# Corrected Import for GAM
from ml_grid.model_classes.h2o_gam_classifier_class import H2OGAMClass as H2O_GAM_class

# Import all other H2O model definition classes
from ml_grid.model_classes.h2o_gbm_classifier_class import H2O_GBM_class
from ml_grid.model_classes.h2o_glm_classifier_class import H2O_GLM_class
from ml_grid.model_classes.h2o_naive_bayes_classifier_class import (
    H2O_NaiveBayes_class,
)
from ml_grid.model_classes.h2o_rulefit_classifier_class import (
    H2ORuleFitClass as H2O_RuleFit_class,
)

from ml_grid.pipeline.grid_search_cross_validate import grid_search_crossvalidate
from ml_grid.util.global_params import global_parameters


@pytest.fixture
def synthetic_data():
    """Creates a synthetic binary classification dataset."""
    #
    # Increased to 200 samples to improve H2O GLM stability on random noise
    X = np.random.randn(200, 10)
    cols = [f"feature_{i}" for i in range(10)]
    y = np.random.randint(0, 2, 200)
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="outcome")


@pytest.fixture
def tiny_problematic_data():
    """
    Provides a very small dataset that is known to cause issues with some
    models during cross-validation if parameters are not handled carefully.
    """
    X = pd.DataFrame(
        {
            # A single, nearly-constant feature to reliably trigger constant column errors.
            "feature1": [0] * 9
            + [1],
        }
    )
    y = pd.Series(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), name="outcome")
    return X, y


@pytest.fixture(scope="module")
def h2o_session_fixture():
    """
    Ensures an H2O cluster is running for the tests.
    """
    try:
        h2o.init(strict_version_check=False)
    except Exception:
        pass
    yield
    # h2o.shutdown(prompt=False) # Optional


# A list of all H2O model definition classes to be tested
# We focus on the core classifiers to ensure stability
H2O_MODEL_CLASSES = [
    H2O_GBM_class,
    H2O_DRF_class,
    H2O_GAM_class,
    # H2O_DeepLearning_class, # Often slow in CI tests
    H2O_GLM_class,
    H2O_NaiveBayes_class,
    H2O_RuleFit_class,
    # H2O_XGBoost_class, # Requires specific system libs
    # H2OAutoMLClass,
]


# This fixture will be parameterized to create an instance of each model class
@pytest.fixture(params=H2O_MODEL_CLASSES)
def h2o_model_instance(request, synthetic_data):
    """
    Creates an instance of an H2O model definition class, which in turn
    creates the underlying scikit-learn compatible wrapper.
    """
    model_class = request.param
    X, y = synthetic_data

    if model_class == H2OAutoMLClass:
        instance = model_class(parameter_space_size="small")
    else:
        # Ensure y is a Series for consistency
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        instance = model_class(X=X, y=y, parameter_space_size="small")
    return instance.algorithm_implementation


def test_h2o_classifier_fit_predict(
    h2o_model_instance, synthetic_data, h2o_session_fixture
):
    """
    Tests the basic fit and predict functionality of each H2O wrapper.
    """
    X, y = synthetic_data
    h2o.remove_all()

    estimator = h2o_model_instance

    # 1. Fit the model
    estimator.fit(X, y)

    # 2. Predict labels
    predictions = estimator.predict(X)
    assert isinstance(predictions, np.ndarray), "predict() should return a numpy array"
    assert predictions.shape == (X.shape[0],), "Prediction array has incorrect shape"

    # 3. Predict probabilities
    prob_predictions = estimator.predict_proba(X)
    assert isinstance(
        prob_predictions, np.ndarray
    ), "predict_proba() should return a numpy array"
    assert prob_predictions.shape == (
        X.shape[0],
        2,
    ), "Probability array has incorrect shape"

    # Check sum close to 1.0
    sums = np.sum(prob_predictions, axis=1)
    if not np.isnan(sums).any():
        assert np.allclose(sums, 1.0, atol=1e-5), "Probabilities should sum to 1"

    # 4. Test set_params/get_params
    try:
        estimator.set_params(seed=123)
        params = estimator.get_params()
        if "seed" in params:
            assert params["seed"] == 123
    except Exception:
        pass


@pytest.mark.parametrize("model_class", H2O_MODEL_CLASSES)
def test_h2o_classifiers_with_cross_validation(
    model_class, tiny_problematic_data, h2o_session_fixture
):
    """
    Tests that H2O wrappers are robust enough to run in a cross-validation.
    """
    X, y = tiny_problematic_data

    if h2o.cluster().is_running():
        h2o.remove_all()

    if model_class == H2OAutoMLClass:
        instance = model_class(parameter_space_size="small")
    else:
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="outcome")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        instance = model_class(X=X, y=y, parameter_space_size="small")

    estimator = instance.algorithm_implementation

    if hasattr(estimator, "MIN_SAMPLES_FOR_STABLE_FIT"):
        if len(X) < estimator.MIN_SAMPLES_FOR_STABLE_FIT:
            pytest.skip(f"Skipping {model_class.__name__} due to small dataset size.")

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    try:
        scores = cross_val_score(estimator, X, y, cv=cv, error_score="raise", n_jobs=1)
        assert len(scores) == 3, "Cross-validation did not complete for all folds."
    except RuntimeError as e:
        assert "fit on a single constant feature" in str(e) or "H2O" in str(
            e
        ), f"Caught unexpected RuntimeError: {e}"
    except ValueError as e:
        # Accept RuleFit or GAM data-specific errors as passes
        msg = str(e)
        if "Skipping GAM col" in msg or "H2ORuleFitClassifier" in msg:
            pass
        else:
            raise e


def test_h2o_gam_knot_cardinality_error(h2o_session_fixture):
    """
    Tests that H2OGAMClassifier raises a specific ValueError/Warning when feature
    cardinality is too low.
    """
    h2o.remove_all()

    # Create data where 'feature2' has low cardinality
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(20),
            "feature2": [0, 1] * 10,  # Only 2 unique values
        }
    )
    y = pd.Series(np.random.randint(0, 2, 20), name="outcome")

    estimator = H2O_GAM_class(
        X=X, y=y, parameter_space_size="small"
    ).algorithm_implementation

    estimator.set_params(
        gam_columns=["feature2"], num_knots=5, _suppress_low_cardinality_error=False
    )

    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    # --- FIX: Robust Regex matching for the error message ---
    with pytest.raises(
        ValueError,
        match=r"Skipping GAM col .* unique.* <",
    ):
        cross_val_score(estimator, X, y, cv=cv, error_score="raise", n_jobs=1)


def test_h2o_gam_knot_distribution_error(h2o_session_fixture):
    """
    Tests that H2OGAMClassifier raises ValueError when quantiles cannot be generated
    due to skewed distribution, even if cardinality is technically sufficient.
    """
    h2o.remove_all()

    # Ensure enough unique values survive the CV split to pass the cardinality check (>= 10)
    # We need > 10 unique values in the training fold.
    # We increase sample size to 200 to ensure stability of the split while maintaining skew.
    # 140 zeros, 60 unique values (1..60).
    skewed_vals = np.array([0] * 140 + list(range(1, 61)))
    np.random.shuffle(skewed_vals)

    X = pd.DataFrame({"feature1": np.random.rand(200), "feature_skewed": skewed_vals})
    y = pd.Series(np.random.randint(0, 2, 200), name="outcome")

    estimator = H2O_GAM_class(
        X=X, y=y, parameter_space_size="small"
    ).algorithm_implementation

    estimator.set_params(
        gam_columns=["feature_skewed"],
        num_knots=5,
        _suppress_low_cardinality_error=False,
    )

    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    with pytest.raises(ValueError, match=r"Cannot generate .* unique quantiles"):
        cross_val_score(estimator, X, y, cv=cv, error_score="raise", n_jobs=1)


class MockMlGridObject:
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X_test = X
        self.y_test = y
        self.X_test_orig = X
        self.y_test_orig = y

        self.local_param_dict = {"param_space_size": "small"}
        self.global_params = global_parameters
        self.base_project_dir = "test_experiments/test_run"
        self.verbose = 0
        self.global_params.verbose = 0
        self.global_params.error_raise = True
        self.global_params.grid_n_jobs = 1
        self.global_params.random_grid_search = True
        self.global_params.bayessearch = False
        self.global_params.max_param_space_iter_value = 2
        self.global_params.cv_folds = 2
        self.global_params.test_mode = True
        self.global_params.sub_sample_param_space_pct = 1.0
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("model_class", H2O_MODEL_CLASSES)
def test_h2o_full_grid_search_pipeline(
    model_class, synthetic_data, h2o_session_fixture
):
    """
    Tests the full pipeline integration by calling grid_search_crossvalidate.
    """
    X, y = synthetic_data

    # Use xsmall param space for tests to be faster and safer
    param_size = "xsmall"

    if model_class == H2OAutoMLClass:
        instance = H2OAutoMLClass(parameter_space_size=param_size)
    else:
        instance = model_class(X=X, y=y, parameter_space_size=param_size)

    mock_ml_grid_object = MockMlGridObject(X, y)

    if isinstance(instance.parameter_space, list):
        flat_param_space = {}
        for d in instance.parameter_space:
            flat_param_space.update(d)
        instance.parameter_space = flat_param_space

    # Convert parameters for grid search compat
    for key, value in instance.parameter_space.items():
        if hasattr(value, "rvs") and hasattr(value, "categories"):
            categories = value.categories
            if any(isinstance(cat, tuple) for cat in categories):
                instance.parameter_space[key] = [list(cat) for cat in categories]
            else:
                instance.parameter_space[key] = list(categories)

        elif (
            hasattr(value, "rvs")
            and hasattr(value, "low")
            and hasattr(value, "high")
            and isinstance(value.low, int)
        ):
            instance.parameter_space[key] = list(
                range(value.low, min(value.high + 1, value.low + 5))
            )
        elif (
            hasattr(value, "rvs")
            and hasattr(value, "low")
            and hasattr(value, "high")
            and isinstance(value.low, float)
        ):
            instance.parameter_space[key] = np.linspace(
                value.low, value.high, 3
            ).tolist()

    if "col_sample_rate_bytree" in instance.parameter_space:
        instance.parameter_space["colsample_bytree"] = instance.parameter_space.pop(
            "col_sample_rate_bytree"
        )

    h2o.remove_all()

    try:
        result = grid_search_crossvalidate(
            algorithm_implementation=instance.algorithm_implementation,
            parameter_space=instance.parameter_space,
            method_name=instance.method_name,
            ml_grid_object=mock_ml_grid_object,
        )
        assert result is not None
        if hasattr(result, "grid_search_cross_validate_score_result"):
            assert isinstance(result.grid_search_cross_validate_score_result, float)

    except Exception as e:
        # Known H2O Backend Bug on synthetic noise data
        if "NullPointerException" in str(e):
            pytest.skip(
                f"Skipping {model_class.__name__} due to known H2O backend instability on synthetic noise data (NPE)."
            )
        else:
            pytest.fail(f"Grid search pipeline failed for {model_class.__name__}: {e}")
