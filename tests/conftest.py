# tests/conftest.py

import logging
import os

import numpy as np
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


@pytest.fixture(params=[False, True], ids=["GridSearch", "BayesSearch"])
def mock_ml_grid_object(request):
    """
    Creates a mock ml_grid_object for both grid and bayes search modes.
    This fixture provides a consistent, synthetic data environment for testing models.
    """
    # Create synthetic data: (n_samples, n_dimensions, n_timesteps)
    X_train = np.random.rand(20, 2, 15)  # n_timesteps=15 to be safe for kernel sizes
    y_train = np.random.randint(0, 2, 20)

    class MockLogger:
        def info(self, msg):
            print(msg)

        def warning(self, msg):
            print(msg)

        def debug(self, msg):
            print(msg)

        def error(self, msg, exc_info=False):
            print(msg)

        def critical(self, msg):
            print(msg)

    class MockGlobalParams:
        def __init__(self, is_bayes):
            self.random_state_val = 42
            self.n_jobs_model_val = 1
            self.time_limit_param = 10
            self.bayessearch = is_bayes
            self.verbose = 0
            self.knn_n_jobs = 1

    class MockPipe:
        def __init__(self, is_bayes):
            self.X_train = X_train
            self.y_train = y_train
            self.global_params = MockGlobalParams(is_bayes)
            self.local_param_dict = {"param_space_size": "small", "n_iter": 2}
            self.verbose = 0
            self.logger = MockLogger()

    return MockPipe(request.param)


@pytest.fixture(params=[False, True], ids=["GridSearch", "BayesSearch"])
def mock_ml_grid_object_standard(request):
    """
    Creates a mock ml_grid_object for standard (non-time-series) models.
    Data is 2D: (n_samples, n_features).
    """
    # Create synthetic data: (n_samples, n_features)
    X_train = np.random.rand(100, 20)
    y_train = np.random.randint(0, 2, 100)

    class MockLogger:
        def info(self, msg):
            print(msg)

        def warning(self, msg):
            print(msg)

        def debug(self, msg):
            print(msg)

        def error(self, msg, exc_info=False):
            print(msg)

        def critical(self, msg):
            print(msg)

    class MockGlobalParams:
        def __init__(self, is_bayes):
            self.random_state_val = 42
            self.n_jobs_model_val = 1
            self.time_limit_param = 10
            self.bayessearch = is_bayes
            self.verbose = 0
            self.knn_n_jobs = 1
            # Standard models might need these
            self.random_grid_search = False
            self.grid_n_jobs = 1
            self.metric_list = ["accuracy"]
            self.error_raise = True
            self.sub_sample_param_space_pct = 1.0
            self.max_param_space_iter_value = 100
            self.n_iter = 2

    class MockPipe:
        def __init__(self, is_bayes):
            self.X_train = X_train
            self.y_train = y_train
            self.global_params = MockGlobalParams(is_bayes)
            self.local_param_dict = {"param_space_size": "small", "n_iter": 2}
            self.verbose = 0
            self.logger = MockLogger()
            self.time_series_mode = False  # Explicitly False

    return MockPipe(request.param)
