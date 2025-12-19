import numpy as np
import pandas as pd
import pytest
import h2o
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_grid.model_classes.H2OGAMClassifier import H2OGAMClassifier


@pytest.fixture(scope="module")
def h2o_session():
    """Ensures H2O is running."""
    try:
        h2o.init(strict_version_check=False)
    except Exception:
        pass
    yield
    # h2o.shutdown(prompt=False)


def create_problematic_data(n_samples=200, n_features=50):
    """
    Generates data that mimics the conditions causing H2O backend crashes:
    1. High dimensionality relative to samples.
    2. Collinear features (simulating what PCA components might look like to H2O).
    3. Sparse-ish distribution.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=10,
        random_state=42,
    )

    # Apply PCA to mimic the user's pipeline state
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # Convert to DataFrame with string columns
    cols = [f"pca_feat_{i}" for i in range(X_pca.shape[1])]
    X_df = pd.DataFrame(X_pca, columns=cols)
    y_series = pd.Series(y, name="outcome")

    return X_df, y_series


def test_gam_fallback_mechanism(h2o_session):
    """
    Tests if H2OGAMClassifier correctly catches the backend error and falls back
    to a stable GLM configuration when passed 'dangerous' parameters.
    """
    h2o.remove_all()
    X, y = create_problematic_data()

    # 1. Setup a GAM classifier with the "Crash" configuration
    # We purposefully ask for parameters that trigger the NPE bug.
    clf = H2OGAMClassifier(
        gam_columns=[X.columns[0]],  # Try to smooth the first PCA component
        num_knots=[5],
        # "Dangerous" settings that usually trigger the Java NPE on this data
        solver="COORDINATE_DESCENT",
        lambda_search=True,
        remove_collinear_columns=True,
    )

    print("\n[TEST] Attempting to fit GAM with unstable parameters...")

    # 2. Fit the model
    # If the fix works, this should NOT crash. It should catch the error internally,
    # log a warning, and succeed using the fallback GLM.
    try:
        clf.fit(X, y)
    except Exception as e:
        pytest.fail(f"GAM Classifier crashed uncaught exception: {e}")

    # 3. Verify the fallback happened
    # We can inspect the internal H2O model parameters to see what actually ran.
    model = h2o.get_model(clf.model_id)

    print(f"[TEST] Final Model Algo: {model.algo}")
    print(f"[TEST] Final Solver Used: {model.actual_params.get('solver')}")

    # Check 1: Did it survive?
    assert clf.model_id is not None

    # Check 2: Did it switch to GLM if GAM failed?
    # (Note: On some stable datasets GAM might actually succeed.
    # If it fails, we want to ensure the resulting model is valid).
    if model.algo == "glm":
        print("[TEST] SUCCESS: Fallback to GLM triggered successfully.")
        # Ensure the fallback used the SAFE solver
        # Note: model.actual_params returns the string value used
        assert (
            model.actual_params["solver"] == "L_BFGS"
        ), "Fallback GLM must use L_BFGS!"
        assert (
            model.actual_params["lambda_search"] is False
        ), "Fallback GLM must disable lambda_search!"


def test_gam_empty_columns_fallback(h2o_session):
    """
    Tests the logic where 'gam_columns' are filtered out entirely (e.g. low cardinality),
    ensuring it switches to GLM instead of crashing with 'Required field gam_columns not specified'.
    """
    h2o.remove_all()
    # Create data with a binary column (cardinality 2)
    X = pd.DataFrame({"binary_col": [0, 1] * 50, "numeric_col": np.random.rand(100)})
    y = pd.Series([0, 1] * 50, name="outcome")

    # Ask to smooth the binary column (which is invalid for GAM knots)
    clf = H2OGAMClassifier(gam_columns=["binary_col"], num_knots=[5])

    print("\n[TEST] Attempting to fit GAM on low-cardinality column...")

    try:
        clf.fit(X, y)
    except Exception as e:
        pytest.fail(f"Classifier crashed on empty gam_columns: {e}")

    model = h2o.get_model(clf.model_id)
    print(f"[TEST] Final Model Algo: {model.algo}")

    # It should have forced a GLM because the only candidate column was rejected
    assert model.algo == "glm"
    print("[TEST] SUCCESS: Correctly fell back to GLM when gam_columns were invalid.")


def test_gam_skewed_distribution_fallback(h2o_session):
    """
    Tests that H2OGAMClassifier detects columns with sufficient cardinality
    but skewed distribution (causing knot generation failure) by falling back
    or raising error as configured.
    """
    h2o.remove_all()

    # Create data: 100 samples.
    # 'feature_skewed': 0 is present 90 times. 1..10 are present 1 time each.
    # Total unique values = 11.
    # If num_knots=5, required unique >= 10. This passes the simple cardinality check.
    # However, quantiles will likely overlap on 0, causing knot generation issues.

    skewed_col = np.array([0] * 90 + list(range(1, 11)))
    np.random.shuffle(skewed_col)

    X = pd.DataFrame({"feature_ok": np.random.rand(100), "feature_skewed": skewed_col})
    y = pd.Series(np.random.randint(0, 2, 100), name="outcome")

    # 1. Test with suppression (default) -> Should drop column and succeed (fallback to GLM if needed)
    clf = H2OGAMClassifier(
        gam_columns=["feature_skewed"],
        num_knots=[5],
        _suppress_low_cardinality_error=True,
    )

    print("\n[TEST] Attempting to fit GAM on skewed column (suppress=True)...")
    try:
        clf.fit(X, y)
        model = h2o.get_model(clf.model_id)
        print(f"[TEST] Model Algo: {model.algo}")
        # Since feature_skewed is the only gam column and it fails knot check,
        # it should be dropped. If no gam cols remain, fallback to GLM.
        assert model.algo == "glm"
    except Exception as e:
        pytest.fail(f"GAM fit failed with suppression enabled: {e}")

    # 2. Test without suppression -> Should raise ValueError from our new check
    clf_raise = H2OGAMClassifier(
        gam_columns=["feature_skewed"],
        num_knots=[5],
        _suppress_low_cardinality_error=False,
    )

    print("\n[TEST] Attempting to fit GAM on skewed column (suppress=False)...")
    with pytest.raises(ValueError, match="Cannot generate .* unique quantiles"):
        clf_raise.fit(X, y)
    print("[TEST] SUCCESS: Caught skewed distribution error.")
