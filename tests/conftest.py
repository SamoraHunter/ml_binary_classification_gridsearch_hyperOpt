# tests/conftest.py

import logging
import os

import h2o
import pytest

# --- Tame TensorFlow ---
# Set log level to suppress info/warnings before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf

    # Explicitly prevent TF from allocating any GPU memory.
    # This stops it from conflicting with H2O's Java VM.
    tf.config.set_visible_devices([], "GPU")
    print("\n--- [Fixture Config] TensorFlow GPU explicitly disabled. ---")
except ImportError:
    print("\n--- [Fixture Config] TensorFlow not found, skipping GPU disable. ---")
    pass
# --- End Tame TensorFlow ---


@pytest.fixture(scope="session")
def h2o_session_fixture():
    """
    Session-scoped fixture to initialize and shut down the H2O cluster.
    This ensures h2o.init() is called only ONCE for the entire test session.
    """
    print("\n--- [H2O Fixture] Initializing H2O cluster... ---")

    # Stop h2o from printing progress bars, which can hang in pytest
    h2o.no_progress()

    # Set up logging
    logging.getLogger("h2o").setLevel(logging.DEBUG)

    try:
        # Start the H2O cluster.
        h2o.init(
            nthreads=-1,  # Use all available cores
            max_mem_size="4g",  # Adjust as needed
            log_level="DEBUG",
        )
        print("--- [H2O Fixture] H2O cluster initialized successfully. ---")

        # Yield to let the tests run
        yield

    finally:
        # This code runs *after* all tests in the session are complete
        print("\n--- [H2O Fixture] Shutting down H2O cluster... ---")

        # Call remove_all() BEFORE shutdown() to avoid ConnectionError
        h2o.remove_all()
        h2o.cluster().shutdown()

        print("--- [H2O Fixture] H2O cluster shutdown complete. ---")


@pytest.fixture(scope="session")
def synthetic_data():
    """Generates simple synthetic data for classification."""
    try:
        from sklearn.datasets import make_classification

        # Keep n_samples large as a safety precaution
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            random_state=42,
        )
        return X, y
    except ImportError:
        pytest.skip("sklearn not installed, skipping synthetic_data generation")
