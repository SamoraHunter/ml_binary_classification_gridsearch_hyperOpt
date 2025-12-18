import logging
import pytest
import h2o
from ml_grid.model_classes.h2o_glm_classifier_class import H2OGLMClassifier
from ml_grid.util.synthetic_data_generator import generate_synthetic_data
from ml_grid.util.impute_data_for_pipe import mean_impute_dataframe

# Setup logger for the test
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="module")
def h2o_session():
    """Ensures H2O is running."""
    try:
        h2o.init(strict_version_check=False)
    except Exception:
        pass
    yield


def test_glm_crash_reproduction(h2o_session):
    """
    Reproduction Test:
    Uses the pipeline's own synthetic data generator to create the exact
    conditions (sparsity + feature structure) that cause the H2O backend crash.
    """
    h2o.remove_all()

    print("\n[TEST] Generating synthetic data using pipeline utilities...")

    # 1. Generate Data
    # We use parameters similar to notebook test
    synthetic_df, important_feature_map = generate_synthetic_data(
        n_rows=200,
        n_features=50,  # Sufficient features to trigger solver complexity
        n_outcome_vars=1,
        feature_strength=0.7,
        percent_important_features=0.2,
        verbose=False,
    )

    # 2. Impute Data (Crucial: Removing NaNs so Python wrapper doesn't block it)
    outcome_columns = list(important_feature_map.keys())
    # We copy to avoid modifying original
    imputed_df = mean_impute_dataframe(data=synthetic_df.copy(), y_vars=outcome_columns)

    # 3. Split X and y
    # The generator creates columns like 'outcome_var_1', we use the first one.
    target_col = outcome_columns[0]
    y = imputed_df[target_col]
    X = imputed_df.drop(columns=outcome_columns)

    # 4. Initialize H2OGLMClassifier with the "Crash Configuration"
    # These are the params that cause the Java NullPointerException
    clf = H2OGLMClassifier(
        solver="COORDINATE_DESCENT",
        remove_collinear_columns=True,
        lambda_search=True,
        family="binomial",
    )

    print(
        f"[TEST] Fitting GLM on Imputed Pipeline Data (Rows={len(X)}, Cols={X.shape[1]})..."
    )

    # 5. Attempt Fit

    try:
        clf.fit(X, y)
    except Exception as e:
        error_msg = str(e)
        if "NullPointerException" in error_msg or "H2OResponseError" in error_msg:
            pytest.fail(f"CRASH REPRODUCED: H2O backend failed.\nError: {e}")
        else:
            pytest.fail(f"Test failed with unexpected error: {e}")

    # 6. Verify Stability & Fix Application
    print("[TEST] Fit completed successfully.")
    assert clf.model_id is not None

    model = h2o.get_model(clf.model_id)
    actual_params = model.actual_params

    print(f"[TEST] Final Solver: {actual_params['solver']}")
    print(f"[TEST] Final Lambda Search: {actual_params['lambda_search']}")

    # Assertions to ensure the fix is actually applied
    if actual_params["lambda_search"] is True:
        pytest.fail("Security Breach: lambda_search was NOT disabled!")

    if actual_params["remove_collinear_columns"] is True:
        pytest.fail("Security Breach: remove_collinear_columns was NOT disabled!")

    # We expect L_BFGS or IRLSM depending on what you hard-locked in the class
    if actual_params["solver"] == "COORDINATE_DESCENT":
        print(
            "WARNING: Solver was not switched. If it didn't crash, you got lucky with the random seed."
        )
